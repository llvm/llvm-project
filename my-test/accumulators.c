// accumulators.c
// 參數：ACC(累加器數), LEN(資料長度), UF(手動unroll倍數)
#ifndef ACC
#define ACC 128
#endif
#ifndef LEN
#define LEN 100000
#endif
#ifndef UF
#define UF 8
#endif

double __attribute__((noinline))
accumulators(double * __restrict a0, double * __restrict a1, double * __restrict out) {
  double s[ACC];
  for (int i=0;i<ACC;i++) s[i]=0.0;

  for (int i=0;i<LEN; i+=UF) {
#pragma clang loop unroll(enable)
    for (int u=0; u<UF; ++u) {
      // 2 個輸入陣列交錯使用，讓use密度更高
      double x = a0[i+u] + a1[i+u];
      // 讓所有累加器都活著：輪流更新不同 s[k]
      for (int k=0;k<ACC;k++) s[k] += x * (k+1);
    }
  }

  double r=0.0;
  for (int k=0;k<ACC;k++) r += s[k]; // 保持 s[] 全程活到尾
  *out = r;
  return r;
}
