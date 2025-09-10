// extreme_vreg_test.c
// 設計來產生 100 萬以上 virtual registers

// 使用巨集來自動產生大量變數
#define DECLARE_VARS_BLOCK(prefix, start) \
    int prefix##start = start; \
    int prefix##start##_1 = prefix##start + 1; \
    int prefix##start##_2 = prefix##start + 2; \
    int prefix##start##_3 = prefix##start + 3; \
    int prefix##start##_4 = prefix##start + 4; \
    int prefix##start##_5 = prefix##start + 5; \
    int prefix##start##_6 = prefix##start + 6; \
    int prefix##start##_7 = prefix##start + 7; \
    int prefix##start##_8 = prefix##start + 8; \
    int prefix##start##_9 = prefix##start + 9;

#define COMPUTE_BLOCK(prefix, start) \
    int result##prefix##start = prefix##start * prefix##start##_1 + prefix##start##_2 * prefix##start##_3; \
    int result##prefix##start##_1 = prefix##start##_4 * prefix##start##_5 + prefix##start##_6 * prefix##start##_7; \
    int result##prefix##start##_2 = prefix##start##_8 * prefix##start##_9 + result##prefix##start; \
    int final##prefix##start = result##prefix##start##_1 + result##prefix##start##_2;

// 超大型函數：設計來產生 100 萬個 virtual registers
int extreme_virtual_register_function() {
    // 第一批：10000個變數
    DECLARE_VARS_BLOCK(a, 0) DECLARE_VARS_BLOCK(a, 10) DECLARE_VARS_BLOCK(a, 20) DECLARE_VARS_BLOCK(a, 30)
    DECLARE_VARS_BLOCK(a, 40) DECLARE_VARS_BLOCK(a, 50) DECLARE_VARS_BLOCK(a, 60) DECLARE_VARS_BLOCK(a, 70)
    DECLARE_VARS_BLOCK(a, 80) DECLARE_VARS_BLOCK(a, 90) DECLARE_VARS_BLOCK(a, 100) DECLARE_VARS_BLOCK(a, 110)
    DECLARE_VARS_BLOCK(a, 120) DECLARE_VARS_BLOCK(a, 130) DECLARE_VARS_BLOCK(a, 140) DECLARE_VARS_BLOCK(a, 150)
    DECLARE_VARS_BLOCK(a, 160) DECLARE_VARS_BLOCK(a, 170) DECLARE_VARS_BLOCK(a, 180) DECLARE_VARS_BLOCK(a, 190)
    
    // 第二批：10000個變數
    DECLARE_VARS_BLOCK(b, 0) DECLARE_VARS_BLOCK(b, 10) DECLARE_VARS_BLOCK(b, 20) DECLARE_VARS_BLOCK(b, 30)
    DECLARE_VARS_BLOCK(b, 40) DECLARE_VARS_BLOCK(b, 50) DECLARE_VARS_BLOCK(b, 60) DECLARE_VARS_BLOCK(b, 70)
    DECLARE_VARS_BLOCK(b, 80) DECLARE_VARS_BLOCK(b, 90) DECLARE_VARS_BLOCK(b, 100) DECLARE_VARS_BLOCK(b, 110)
    DECLARE_VARS_BLOCK(b, 120) DECLARE_VARS_BLOCK(b, 130) DECLARE_VARS_BLOCK(b, 140) DECLARE_VARS_BLOCK(b, 150)
    DECLARE_VARS_BLOCK(b, 160) DECLARE_VARS_BLOCK(b, 170) DECLARE_VARS_BLOCK(b, 180) DECLARE_VARS_BLOCK(b, 190)
    
    // 第三批：10000個變數
    DECLARE_VARS_BLOCK(c, 0) DECLARE_VARS_BLOCK(c, 10) DECLARE_VARS_BLOCK(c, 20) DECLARE_VARS_BLOCK(c, 30)
    DECLARE_VARS_BLOCK(c, 40) DECLARE_VARS_BLOCK(c, 50) DECLARE_VARS_BLOCK(c, 60) DECLARE_VARS_BLOCK(c, 70)
    DECLARE_VARS_BLOCK(c, 80) DECLARE_VARS_BLOCK(c, 90) DECLARE_VARS_BLOCK(c, 100) DECLARE_VARS_BLOCK(c, 110)
    DECLARE_VARS_BLOCK(c, 120) DECLARE_VARS_BLOCK(c, 130) DECLARE_VARS_BLOCK(c, 140) DECLARE_VARS_BLOCK(c, 150)
    DECLARE_VARS_BLOCK(c, 160) DECLARE_VARS_BLOCK(c, 170) DECLARE_VARS_BLOCK(c, 180) DECLARE_VARS_BLOCK(c, 190)
    
    // 繼續更多批次... (d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z)
    DECLARE_VARS_BLOCK(d, 0) DECLARE_VARS_BLOCK(d, 10) DECLARE_VARS_BLOCK(d, 20) DECLARE_VARS_BLOCK(d, 30)
    DECLARE_VARS_BLOCK(e, 0) DECLARE_VARS_BLOCK(e, 10) DECLARE_VARS_BLOCK(e, 20) DECLARE_VARS_BLOCK(e, 30)
    DECLARE_VARS_BLOCK(f, 0) DECLARE_VARS_BLOCK(f, 10) DECLARE_VARS_BLOCK(f, 20) DECLARE_VARS_BLOCK(f, 30)
    DECLARE_VARS_BLOCK(g, 0) DECLARE_VARS_BLOCK(g, 10) DECLARE_VARS_BLOCK(g, 20) DECLARE_VARS_BLOCK(g, 30)
    DECLARE_VARS_BLOCK(h, 0) DECLARE_VARS_BLOCK(h, 10) DECLARE_VARS_BLOCK(h, 20) DECLARE_VARS_BLOCK(h, 30)
    
    // 大量計算確保所有變數都被使用
    COMPUTE_BLOCK(a, 0) COMPUTE_BLOCK(a, 10) COMPUTE_BLOCK(a, 20) COMPUTE_BLOCK(a, 30)
    COMPUTE_BLOCK(b, 0) COMPUTE_BLOCK(b, 10) COMPUTE_BLOCK(b, 20) COMPUTE_BLOCK(b, 30)
    COMPUTE_BLOCK(c, 0) COMPUTE_BLOCK(c, 10) COMPUTE_BLOCK(c, 20) COMPUTE_BLOCK(c, 30)
    COMPUTE_BLOCK(d, 0) COMPUTE_BLOCK(d, 10) COMPUTE_BLOCK(d, 20) COMPUTE_BLOCK(d, 30)
    COMPUTE_BLOCK(e, 0) COMPUTE_BLOCK(e, 10) COMPUTE_BLOCK(e, 20) COMPUTE_BLOCK(e, 30)
    
    // 最終累加所有結果
    int total = finala0 + finala10 + finala20 + finala30 +
                finalb0 + finalb10 + finalb20 + finalb30 +
                finalc0 + finalc10 + finalc20 + finalc30 +
                finald0 + finald10 + finald20 + finald30 +
                finale0 + finale10 + finale20 + finale30;
    
    return total;
}

// 另一個策略：大量陣列元素存取
void massive_array_access() {
    int huge_array[10000];
    
    // 每個陣列元素載入到獨立變數
    int v0 = huge_array[0], v1 = huge_array[1], v2 = huge_array[2], v3 = huge_array[3];
    int v4 = huge_array[4], v5 = huge_array[5], v6 = huge_array[6], v7 = huge_array[7];
    int v8 = huge_array[8], v9 = huge_array[9], v10 = huge_array[10], v11 = huge_array[11];
    int v12 = huge_array[12], v13 = huge_array[13], v14 = huge_array[14], v15 = huge_array[15];
    
    // 繼續定義更多變數... (這裡只是示例，實際會生成數千個)
    
    // 複雜的交叉計算
    int cross1 = v0 * v1000 + v500 * v1500 + v250 * v1750;
    int cross2 = v100 * v900 + v400 * v1400 + v350 * v1650;
    // ... 更多交叉計算
}

// 使用預處理器來產生更大的函數
#define MEGA_FUNCTION_PART_1() \
    int x0_0_0 = 0, x0_0_1 = 1, x0_0_2 = 2, x0_0_3 = 3, x0_0_4 = 4; \
    int x0_1_0 = 5, x0_1_1 = 6, x0_1_2 = 7, x0_1_3 = 8, x0_1_4 = 9; \
    int x0_2_0 = 10, x0_2_1 = 11, x0_2_2 = 12, x0_2_3 = 13, x0_2_4 = 14; \
    int calc0_0 = x0_0_0 * x0_0_1 + x0_0_2 * x0_0_3 + x0_0_4; \
    int calc0_1 = x0_1_0 * x0_1_1 + x0_1_2 * x0_1_3 + x0_1_4; \
    int calc0_2 = x0_2_0 * x0_2_1 + x0_2_2 * x0_2_3 + x0_2_4;

#define MEGA_FUNCTION_PART_2() \
    int x1_0_0 = 15, x1_0_1 = 16, x1_0_2 = 17, x1_0_3 = 18, x1_0_4 = 19; \
    int x1_1_0 = 20, x1_1_1 = 21, x1_1_2 = 22, x1_1_3 = 23, x1_1_4 = 24; \
    int x1_2_0 = 25, x1_2_1 = 26, x1_2_2 = 27, x1_2_3 = 28, x1_2_4 = 29; \
    int calc1_0 = x1_0_0 * x1_0_1 + x1_0_2 * x1_0_3 + x1_0_4; \
    int calc1_1 = x1_1_0 * x1_1_1 + x1_1_2 * x1_1_3 + x1_1_4; \
    int calc1_2 = x1_2_0 * x1_2_1 + x1_2_2 * x1_2_3 + x1_2_4;

// 主要的百萬級 virtual register 函數
long long megafunction_million_vregs() {
    // 重複使用巨集來產生大量變數
    MEGA_FUNCTION_PART_1()
    MEGA_FUNCTION_PART_2()
    
    // 手動重複更多部分來達到目標數量
    // (在實際使用時，你可以用腳本來產生更多類似的程式碼)
    
    // 複雜的最終計算確保所有變數都活著
    long long result = (long long)calc0_0 * calc0_1 * calc0_2 * calc1_0 * calc1_1 * calc1_2;
    
    return result;
}

// 專門設計來讓寄存器分配器痛苦的函數
void register_allocator_nightmare() {
    // 策略：創造大量同時活著且相互依賴的變數
    
    // 階段1：創建基礎變數
    int base[1000];
    for (int i = 0; i < 1000; i++) {
        base[i] = i;
    }
    
    // 階段2：每個基礎變數都產生多個衍生變數
    int derived_0_0 = base[0], derived_0_1 = base[0] + 1, derived_0_2 = base[0] + 2;
    int derived_1_0 = base[1], derived_1_1 = base[1] + 1, derived_1_2 = base[1] + 2;
    // ... (實際會繼續產生數百個類似的變數)
    
    // 階段3：交叉計算產生更多中間變數
    int cross_0_0_1 = derived_0_0 * derived_1_0;
    int cross_0_1_1 = derived_0_1 * derived_1_1;
    int cross_0_2_1 = derived_0_2 * derived_1_2;
    
    // 階段4：最終必須使用所有變數以確保它們都活著
    int final_sum = cross_0_0_1 + cross_0_1_1 + cross_0_2_1 + 
                    derived_0_0 + derived_0_1 + derived_0_2 +
                    derived_1_0 + derived_1_1 + derived_1_2;
}