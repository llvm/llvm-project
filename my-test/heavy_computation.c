// 測試案例 1: 大量區域變數 + 複雜計算
// 這會產生高寄存器壓力，迫使演算法做出 spill 決策
int heavy_computation(int a, int b, int c, int d) {
    int v1 = a * b + c;
    int v2 = b * c + d;
    int v3 = c * d + a;
    int v4 = d * a + b;
    int v5 = v1 * v2;
    int v6 = v2 * v3;
    int v7 = v3 * v4;
    int v8 = v4 * v1;
    int v9 = v5 + v6;
    int v10 = v6 + v7;
    int v11 = v7 + v8;
    int v12 = v8 + v5;
    int v13 = v9 * v10;
    int v14 = v10 * v11;
    int v15 = v11 * v12;
    int v16 = v12 * v9;
    
    // 強迫所有變數都還活著
    return v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + 
           v9 + v10 + v11 + v12 + v13 + v14 + v15 + v16;
}
