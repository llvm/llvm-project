// Control dependency
void control_dep(int *A, int n, int x) {
  for (int i = 0; i < n; ++i) {
    if (x > 0) {        // 控制條件
      A[i] = A[i] * 2;  // 這個操作依賴於上面的條件
    }
  }
}
