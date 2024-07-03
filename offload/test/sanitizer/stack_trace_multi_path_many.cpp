// clang-format off
// : %libomptarget-compileoptxx-generic -fsanitize=offload -O1
// : not %libomptarget-run-generic 2> %t.out
// : %fcheck-generic --check-prefixes=CHECK < %t.out
// : %libomptarget-compileoptxx-generic -fsanitize=offload -O3
// : not %libomptarget-run-generic 2> %t.out
// RUN: %libomptarget-compileoptxx-generic -fsanitize=offload -O3 -g
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=DEBUG < %t.out
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

template <typename T> [[clang::optnone]] T deref(T *P) { return *P; }

template <int LEVEL, typename T> [[gnu::noinline]] T level(T *P) {
  if constexpr (LEVEL > 1)
    return level<LEVEL - 1>(P) + level<LEVEL - 2>(P);
  if constexpr (LEVEL > 0)
    return level<LEVEL - 1>(P);
  return deref(P);
}

int main(void) {

  int *ValidInt = (int *)omp_target_alloc(4, omp_get_default_device());
#pragma omp target is_device_ptr(ValidInt)
  {
    level<12>(ValidInt);
    short *ValidShort = ((short *)ValidInt) + 2;
    level<12>(ValidShort);
    char *Invalid = ((char *)ValidInt) + 4;
    level<12>(Invalid);
  }
}
