//-----------------------------------------------------------------------------
// Definitions.
//-----------------------------------------------------------------------------
#ifndef SUITE_DEFINITIONS_H
#define SUITE_DEFINITIONS_H

#ifdef _MSC_VER
#define forceinline __forceinline
#define OPTIMIZE_OFF __pragma(optimize("", off))
#define OPTIMIZE_ON  __pragma(optimize("", on))
#elif defined(__clang__)
#if __has_attribute(__always_inline__)
#define forceinline inline __attribute__((__always_inline__))
#else
#define forceinline inline
#endif
#define OPTIMIZE_OFF _Pragma("clang optimize off")
#define OPTIMIZE_ON  _Pragma("clang optimize on")
#elif defined(__GNUC__)
#define forceinline inline __attribute__((__always_inline__))
#define OPTIMIZE_OFF _Pragma("GCC optimize off")
#define OPTIMIZE_ON  _Pragma("GCC optimize on")
#else
#define forceinline inline
#define OPTIMIZE_OFF
#define OPTIMIZE_ON
#error
#endif

#endif // SUITE_DEFINITIONS_H
