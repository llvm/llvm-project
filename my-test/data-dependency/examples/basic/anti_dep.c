// Anti dependency (WAR - Write After Read)
void anti_dep(int *A, int n) {
  for (int i = 0; i < n-1; ++i) {
    int temp = A[i+1];   // 讀 A[i+1]
    A[i] = temp + 1;     // 寫 A[i]  → A[i+1] 被後面讀取時產生 WAR
  }
}
