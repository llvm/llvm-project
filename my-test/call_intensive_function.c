// 測試案例 3: 函數呼叫密集 + 長生存期變數
// 測試跨函數呼叫的寄存器保存策略
int call_intensive_function(int x) {
    int long_lived1 = x * 2;
    int long_lived2 = x * 3;
    int long_lived3 = x * 5;
    int long_lived4 = x * 7;
    
    // 大量函數呼叫會影響寄存器使用
    int result1 = heavy_computation(x, long_lived1, long_lived2, long_lived3);
    int result2 = heavy_computation(long_lived1, long_lived2, long_lived3, long_lived4);
    int result3 = heavy_computation(long_lived2, long_lived3, long_lived4, x);
    int result4 = heavy_computation(long_lived3, long_lived4, x, long_lived1);
    
    // 這些變數必須在函數呼叫後仍然活著
    return long_lived1 + long_lived2 + long_lived3 + long_lived4 +
           result1 + result2 + result3 + result4;
}
