// 測試案例 4: 複雜的條件分支 + 大量暫存變數
// 測試在不同控制流路徑中的寄存器分配
int complex_branching(int a, int b, int c, int d, int e, int f) {
    int x1 = a + b;
    int x2 = c + d;
    int x3 = e + f;
    int x4 = a * b;
    int x5 = c * d;
    int x6 = e * f;
    
    if (x1 > x2) {
        int y1 = x1 * x3;
        int y2 = x2 * x4;
        int y3 = x3 * x5;
        int y4 = x4 * x6;
        
        if (y1 > y2) {
            int z1 = y1 + y3;
            int z2 = y2 + y4;
            int z3 = y3 + x1;
            int z4 = y4 + x2;
            return z1 * z2 + z3 * z4 + x1 + x2 + x3 + x4 + x5 + x6;
        } else {
            int z1 = y2 + y4;
            int z2 = y1 + y3;
            int z3 = y4 + x3;
            int z4 = y3 + x4;
            return z1 * z2 + z3 * z4 + x1 + x2 + x3 + x4 + x5 + x6;
        }
    } else {
        int y1 = x2 * x6;
        int y2 = x1 * x5;
        int y3 = x6 * x4;
        int y4 = x5 * x3;
        
        if (y1 < y2) {
            int z1 = y1 - y3;
            int z2 = y2 - y4;
            int z3 = y3 - x5;
            int z4 = y4 - x6;
            return z1 * z2 - z3 * z4 + x1 + x2 + x3 + x4 + x5 + x6;
        } else {
            int z1 = y2 - y4;
            int z2 = y1 - y3;
            int z3 = y4 - x1;
            int z4 = y3 - x2;
            return z1 * z2 - z3 * z4 + x1 + x2 + x3 + x4 + x5 + x6;
        }
    }
}
