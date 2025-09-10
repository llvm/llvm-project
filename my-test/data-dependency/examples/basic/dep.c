// file: dep.c
void g(int *A, int n) {
  for (int i = 1; i < n; ++i) {
    A[i] = A[i-1] + 1;   // 讀 A[i-1]，寫 A[i]  → 典型 flow (RAW) 相依
  }
}
