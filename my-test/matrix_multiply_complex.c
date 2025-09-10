// 測試案例 5: 矩陣運算 - 真實世界的高寄存器壓力場景
void matrix_multiply_complex(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            double temp1 = A[i * n + 0] * B[0 * n + j];
            double temp2 = A[i * n + 1] * B[1 * n + j];
            double temp3 = A[i * n + 2] * B[2 * n + j];
            double temp4 = A[i * n + 3] * B[3 * n + j];
            
            // 手動展開迴圈增加寄存器壓力
            for (int k = 4; k < n; k += 4) {
                double a1 = A[i * n + k];
                double a2 = A[i * n + k + 1];
                double a3 = A[i * n + k + 2];
                double a4 = A[i * n + k + 3];
                
                double b1 = B[k * n + j];
                double b2 = B[(k + 1) * n + j];
                double b3 = B[(k + 2) * n + j];
                double b4 = B[(k + 3) * n + j];
                
                temp1 += a1 * b1;
                temp2 += a2 * b2;
                temp3 += a3 * b3;
                temp4 += a4 * b4;
                
                sum += temp1 + temp2 + temp3 + temp4;
            }
            
            C[i * n + j] = sum + temp1 + temp2 + temp3 + temp4;
        }
    }
}