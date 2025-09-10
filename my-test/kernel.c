// K: 同時存活度；B: 每次迭代產生的暫時值數；R: 回合數
#ifndef K
#define K 28
#endif
#ifndef B
#define B 3
#endif
#ifndef R
#define R 4096
#endif

static inline float touch(float x) { asm volatile("" :: "w"(x)); return x; }

float kernel(const float *a) __attribute__((noinline));
float kernel(const float *a) {
  float acc[K]; for (int i=0;i<K;++i) acc[i]=0.0f;

  int idx=0;
  for (int r=0;r<R;++r) {
#pragma clang loop unroll(disable)
    for (int i=0;i<K;++i) {
      float t0 = a[idx++] + acc[i];
      float t1 = t0 * 1.0001f;
      float t2 = t1 + 3.14f;     // B=3 個暫時值同時活躍到此點
      acc[i] = acc[i] + touch(t2); // 延長壽命，避免 DCE/提早釋放
    }
  }
  float s=0; for (int i=0;i<K;++i) s+=acc[i]; return s;
}
