// RUN: %libomp-compile
// RUN: env OMP_DYNAMIC=FALSE KMP_INHERIT_FP_CONTROL=TRUE %libomp-run
// REQUIRES: linux, riscv64-target-arch, riscv-hardware-fp
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fenv.h>
#include <omp.h>
#include <stdio.h>

// Reused workers must inherit the new rounding mode at each parallel region.
// Inspect control state directly, without relying on arithmetic optimization.
int main(void) {
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  int original = fegetround();
  int errors = 0;
  omp_set_dynamic(0);
  for (int stage = 0; stage < 4; ++stage) {
    if (fesetround(modes[stage]))
      return 2;
    int bad = 0;
#pragma omp parallel num_threads(2) reduction(+:bad)
    {
      if (omp_get_num_threads() != 2 || fegetround() != modes[stage])
        ++bad;
    }
    printf("FP_ROUND_STAGE stage=%d errors=%d\n", stage, bad);
    errors += bad;
  }
  if (fesetround(original))
    return 2;
  printf("FP_ROUND_RESULT errors=%d\n", errors);
  return errors != 0;
}
