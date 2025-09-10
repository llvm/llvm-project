// Output dependency (WAW - Write After Write)
void output_dep(int *A, int n) {
  for (int i = 0; i < n; ++i) {
    A[i] = i;           // 第一次寫 A[i]
    if (i % 2 == 0) {
      A[i] = i * 2;     // 第二次寫 A[i] → WAW dependency
    }
  }
}
