// file: neon_window.c
// AArch64 / NEON 相位式視窗壓力測試（可用 -O3）
#include <arm_neon.h>
#include <stddef.h>

#ifndef W
#define W 28   // 視窗內同時存活的向量寄存器數目（調大以加壓，<= 28 比較實際）
#endif

#ifndef R
#define R 8192 // 迴圈輪數（調大以拉長編譯器分析/排程負載）
#endif

#ifndef U
#define U 16    // 展開因子（放大相位變化的批量更新）
#endif

static inline float32x4_t touch(float32x4_t x) {
  // 防止被最佳化掉：引入輕微的 data dependence
  asm volatile("" :: "w"(x));
  return x;
}

float sum_windows(const float* __restrict a, size_t n) {
  float32x4_t win[W];
  for (int i = 0; i < W; ++i) win[i] = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (int r = 0; r < R; ++r) {
#pragma clang loop unroll_count(U)
    for (int u = 0; u < U; ++u) {
      // 批量「退場 + 進場」：形成滑動視窗的相位式生命週期
      int drop = (r*U + u) % W;
      win[drop] = vdupq_n_f32(0.0f); // 釋放舊成員（邏輯上）
      // 新成員進場：依序讀取
      if (i + 4 <= n) {
        float32x4_t v = vld1q_f32(&a[i]);
        win[drop] = vaddq_f32(win[drop], v);
        i += 4;
      }
      // 觸碰所有活躍者，強化它們的存活區間（維持壓力）
      for (int k = 0; k < W; ++k) win[k] = touch(win[k]);
    }
  }

  // 匯總，避免被最佳化整體刪除
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int k = 0; k < W; ++k) acc = vaddq_f32(acc, win[k]);
  float out[4];
  vst1q_f32(out, acc);
  return out[0] + out[1] + out[2] + out[3];
}
