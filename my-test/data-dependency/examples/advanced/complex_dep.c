// 複雜的記憶體相依模式
void complex_dep(int *A, int *B, int n) {
  // 多個陣列之間的相依
  for (int i = 1; i < n; ++i) {
    A[i] = A[i-1] + B[i];     // RAW on A[i-1]
    B[i+1] = A[i] * 2;        // RAW on A[i], 但寫入 B[i+1]
  }
}

void indirect_dep(int *A, int *idx, int n) {
  // 間接記憶體存取的相依
  for (int i = 0; i < n; ++i) {
    A[idx[i]] = A[idx[i]] + 1;  // 透過 idx[i] 間接存取
  }
}

void aliasing_dep(int *A, int *B, int n) {
  // 可能的 aliasing 問題
  for (int i = 0; i < n; ++i) {
    *A = *B + 1;              // 如果 A 和 B 指向相同或重疊記憶體
    B++;
    A++;
  }
}
