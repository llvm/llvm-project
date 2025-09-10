// 可向量化的循環 (無相依)
void vectorizable(int *A, int *B, int n) {
  for (int i = 0; i < n; ++i) {
    A[i] = B[i] * 2;          // 每個迭代獨立
  }
}

// 不可向量化的循環 (有相依)
void non_vectorizable(int *A, int n) {
  for (int i = 1; i < n; ++i) {
    A[i] = A[i-1] + A[i];     // 相依距離為 1
  }
}

// 可向量化的循環 (相依距離 > 向量長度)
void distant_dep(int *A, int n) {
  for (int i = 4; i < n; ++i) {
    A[i] = A[i-4] + 1;        // 相依距離為 4，可能可以向量化
  }
}
