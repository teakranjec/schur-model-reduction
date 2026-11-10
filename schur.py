
import numpy as np
from scipy.linalg import (
    schur,
    svd,
    eigvals,
    solve_continuous_lyapunov
)


def schurova_metoda(A, B, C, D, k):
    """
    Schurova metoda balansirane redukcije modela.

    Parametri:
        A - matrica sustava (n x n)
        B - matrica ulaza (n x m)
        C - matrica izlaza (p x n)
        D - matrica izravnog prijenosa (p x m)
        k - zeljeni red reduciranog modela

    Vraca:
        Ar, Br, Cr, Dr - matrice reduciranog sustava
        hsv - Hankelove singularne vrijednosti
        SL, SR - lijeva i desna projekcijska matrica
    """

    # 1. Pretvorba ulaznih podataka u matrice
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)

    if any(M.ndim != 2 for M in (A, B, C, D)):
        raise ValueError("Sve matrice moraju biti dvodimenzionalne.")

    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("Matrica A mora biti kvadratna.")

    if B.shape[0] != n or C.shape[1] != n:
        raise ValueError("Dimenzije matrica A, B i C nisu kompatibilne.")

    if D.shape != (C.shape[0], B.shape[1]):
        raise ValueError("Dimenzije matrice D nisu kompatibilne.")

    if not isinstance(k, (int, np.integer)) or not 1 <= k < n:
        raise ValueError("Mora vrijediti 1 <= k < n.")

    # 2. Provjera stabilnosti izvornog sustava
    if np.any(np.real(eigvals(A)) >= 0):
        raise ValueError("Izvorni sustav mora biti asimptotski stabilan.")

    # 3. Racunanje Gramianovih matrica
    # A P + P A^T + B B^T = 0
    # A^T Q + Q A + C^T C = 0

    P = solve_continuous_lyapunov(A, -(B @ B.T))
    Q = solve_continuous_lyapunov(A.T, -(C.T @ C))

    # Uklanjanje malih numerickih odstupanja od simetrije
    P = (P + P.T) / 2
    Q = (Q + Q.T) / 2

    # 4. Produkti Gramianovih matrica
    PQ = P @ Q
    QP = Q @ P

    # 5. Hankelove singularne vrijednosti
    lam = np.sort(np.real(eigvals(PQ)))[::-1]

    if lam[0] <= 0:
        raise ValueError("Sustav nema pozitivnih Hankelovih vrijednosti.")

    if np.min(lam) < -1e-8 * max(1.0, lam[0]):
        raise ValueError("Produkt PQ ima znacajno negativne svojstvene vrijednosti.")

    hsv = np.sqrt(np.maximum(lam, 0.0))

    # 6. Granica za izdvajanje dominantnih potprostora
    tolerancija = 1e-12 * max(1.0, abs(lam[0]))

    if lam[k - 1] <= tolerancija:
        raise ValueError("Odabrani red ukljucuje numericki zanemarive modove.")

    if lam[k - 1] - lam[k] <= tolerancija:
        raise ValueError(
            "Dominantne svojstvene vrijednosti nisu dovoljno razdvojene."
        )

    granica = (lam[k - 1] + lam[k]) / 2

    # 7. Uredene Schurove dekompozicije
    # SciPy za realnu Schurovu formu koristi wr i wi
    TA, VA, broj_A = schur(
        PQ,
        output="real",
        sort=lambda wr, wi: wr > granica
    )

    TD, VD, broj_D = schur(
        QP,
        output="real",
        sort=lambda wr, wi: wr > granica
    )

    if broj_A != k or broj_D != k:
        raise ValueError("Nije moguce izdvojiti trazene Schurove potprostore.")

    # 8. Dominantni desni i lijevi potprostori
    VR = VA[:, :k]
    VL = VD[:, :k]

    # 9. Matrica E i singularna dekompozicija
    E = VL.T @ VR

    U, s, Vt = svd(E)

    if s[-1] <= 1e-12 * s[0]:
        raise ValueError("Matrica E je numericki singularna.")

    # 10. Konstrukcija biortogonalnih projekcijskih matrica
    SL = (VL @ U) / np.sqrt(s)[None, :]
    SR = (VR @ Vt.T) / np.sqrt(s)[None, :]

    # 11. Konstrukcija reduciranog modela
    Ar = SL.T @ A @ SR
    Br = SL.T @ B
    Cr = C @ SR
    Dr = D.copy()

    return Ar, Br, Cr, Dr, hsv, SL, SR
