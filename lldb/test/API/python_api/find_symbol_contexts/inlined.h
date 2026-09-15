#ifndef INLINED_H
#define INLINED_H

#if defined(__clang__)
#define ALWAYS_INLINE [[clang::always_inline]] inline
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ALWAYS_INLINE inline
#endif

ALWAYS_INLINE int inlined_add(int a, int b) {
  int sum = a + b; // inlined body
  return sum;
}

#endif
