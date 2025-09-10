// 教學用：詳細註解的 data dependency 範例

/* 
 * RAW (Read After Write) - Flow Dependency
 * 最常見的相依性，後面的指令讀取前面指令寫入的值
 */
void raw_example(int *A, int n) {
  for (int i = 1; i < n; ++i) {
    A[i] = A[i-1] + 1;   // S1: 讀 A[i-1] (前一個迭代寫入)
                         // S2: 寫 A[i]
                         // 迭代 k+1 的 S1 依賴迭代 k 的 S2
  }
}

/* 
 * WAR (Write After Read) - Anti Dependency
 * 後面的寫入不能在前面的讀取完成前執行
 */
void war_example(int *A, int n) {
  for (int i = 0; i < n-1; ++i) {
    int temp = A[i+1];   // S1: 讀 A[i+1]
    A[i] = temp + 1;     // S2: 寫 A[i]
                         // 如果重排序，S2 可能會影響下個迭代的 S1
  }
}

/* 
 * WAW (Write After Write) - Output Dependency
 * 兩個寫入操作的順序必須保持
 */
void waw_example(int *A, int n) {
  for (int i = 0; i < n; ++i) {
    A[i] = i;           // S1: 寫 A[i]
    A[i] = i * 2;       // S2: 寫 A[i]
                        // S2 必須在 S1 之後，否則結果錯誤
  }
}
