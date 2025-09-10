// file: pressure_block.c
#include <arm_neon.h>
#ifndef W
#define W 28  // 盡量接近 AArch64 向量暫存器上限
#endif
#ifndef R
#define R 2048 // 放大 IR/排程負擔（不是關鍵，但有幫助）
#endif

// 防止被 DCE：把所有暫存器在最後一次性歸約
static inline float reduce_all(float32x4_t *v, int n) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (int i = 0; i < n; ++i) acc = vaddq_f32(acc, v[i]);
  float out[4];
  vst1q_f32(out, acc);
  return out[0] + out[1] + out[2] + out[3];
}

float kernel(const float *a) __attribute__((noinline));
float kernel(const float *a) {
  // 明確建 W 個獨立向量變數，確保它們同時活到結尾
  float32x4_t v[W];
#pragma clang loop unroll(disable)
  for (int i = 0; i < W; ++i) v[i] = vdupq_n_f32(0.0f);

  int idx = 0;
  for (int r = 0; r < R; ++r) {
    // 在同一個基本區塊內，製造大量同時存活的臨時值
#pragma clang loop unroll(disable)
    for (int i = 0; i < W; ++i) {
      float32x4_t t0 = vld1q_f32(&a[idx]);
      float32x4_t t1 = vaddq_f32(t0, v[i]);
      float32x4_t t2 = vmulq_f32(t1, vdupq_n_f32(1.0001f));
      v[i] = vaddq_f32(v[i], t2);   // 讓 t0,t1,t2 在此點前都同時存活
      idx += 4;
      asm volatile("" :: "w"(v[i]) : "memory"); // 防止重排/消除
    }
  }
  return reduce_all(v, W);
}
