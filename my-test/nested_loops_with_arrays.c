// 測試案例 2: 多層巢狀迴圈 + 陣列存取
// 測試在迴圈中的 spill 成本權重
void nested_loops_with_arrays(int *arr1, int *arr2, int *result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                int temp1 = arr1[i * size + j];
                int temp2 = arr2[j * size + k];
                int temp3 = temp1 * temp2;
                int temp4 = temp3 + arr1[k];
                int temp5 = temp4 * arr2[i];
                int temp6 = temp5 + temp1;
                int temp7 = temp6 * temp2;
                int temp8 = temp7 + temp3;
                
                result[i * size * size + j * size + k] = 
                    temp1 + temp2 + temp3 + temp4 + 
                    temp5 + temp6 + temp7 + temp8;
            }
        }
    }
}
