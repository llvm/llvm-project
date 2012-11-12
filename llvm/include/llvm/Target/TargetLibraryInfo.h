//===-- llvm/Target/TargetLibraryInfo.h - Library information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETLIBRARYINFO_H
#define LLVM_TARGET_TARGETLIBRARYINFO_H

#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
  class Triple;

  namespace LibFunc {
    enum Func {
      /// void operator delete[](void*);
      ZdaPv,
      /// void operator delete(void*);
      ZdlPv,
      /// void *new[](unsigned int);
      Znaj,
      /// void *new[](unsigned int, nothrow);
      ZnajRKSt9nothrow_t,
      /// void *new[](unsigned long);
      Znam,
      /// void *new[](unsigned long, nothrow);
      ZnamRKSt9nothrow_t,
      /// void *new(unsigned int);
      Znwj,
      /// void *new(unsigned int, nothrow);
      ZnwjRKSt9nothrow_t,
      /// void *new(unsigned long);
      Znwm,
      /// void *new(unsigned long, nothrow);
      ZnwmRKSt9nothrow_t,
      /// int __cxa_atexit(void (*f)(void *), void *p, void *d);
      cxa_atexit,
      /// void __cxa_guard_abort(guard_t *guard);
      /// guard_t is int64_t in Itanium ABI or int32_t on ARM eabi.
      cxa_guard_abort,      
      /// int __cxa_guard_acquire(guard_t *guard);
      cxa_guard_acquire,
      /// void __cxa_guard_release(guard_t *guard);
      cxa_guard_release,
      /// void *__memcpy_chk(void *s1, const void *s2, size_t n, size_t s1size);
      memcpy_chk,
      /// double acos(double x);
      acos,
      /// float acosf(float x);
      acosf,
      /// double acosh(double x);
      acosh,
      /// float acoshf(float x);
      acoshf,
      /// long double acoshl(long double x);
      acoshl,
      /// long double acosl(long double x);
      acosl,
      /// double asin(double x);
      asin,
      /// float asinf(float x);
      asinf,
      /// double asinh(double x);
      asinh,
      /// float asinhf(float x);
      asinhf,
      /// long double asinhl(long double x);
      asinhl,
      /// long double asinl(long double x);
      asinl,
      /// double atan(double x);
      atan,
      /// double atan2(double y, double x);
      atan2,
      /// float atan2f(float y, float x);
      atan2f,
      /// long double atan2l(long double y, long double x);
      atan2l,
      /// float atanf(float x);
      atanf,
      /// double atanh(double x);
      atanh,
      /// float atanhf(float x);
      atanhf,
      /// long double atanhl(long double x);
      atanhl,
      /// long double atanl(long double x);
      atanl,
      /// void *calloc(size_t count, size_t size);
      calloc,
      /// double cbrt(double x);
      cbrt,
      /// float cbrtf(float x);
      cbrtf,
      /// long double cbrtl(long double x);
      cbrtl,
      /// double ceil(double x);
      ceil,
      /// float ceilf(float x);
      ceilf,
      /// long double ceill(long double x);
      ceill,
      /// double copysign(double x, double y);
      copysign,
      /// float copysignf(float x, float y);
      copysignf,
      /// long double copysignl(long double x, long double y);
      copysignl,
      /// double cos(double x);
      cos,
      /// float cosf(float x);
      cosf,
      /// double cosh(double x);
      cosh,
      /// float coshf(float x);
      coshf,
      /// long double coshl(long double x);
      coshl,
      /// long double cosl(long double x);
      cosl,
      /// double exp(double x);
      exp,
      /// double exp10(double x);
      exp10,
      /// float exp10f(float x);
      exp10f,
      /// long double exp10l(long double x);
      exp10l,
      /// double exp2(double x);
      exp2,
      /// float exp2f(float x);
      exp2f,
      /// long double exp2l(long double x);
      exp2l,
      /// float expf(float x);
      expf,
      /// long double expl(long double x);
      expl,
      /// double expm1(double x);
      expm1,
      /// float expm1f(float x);
      expm1f,
      /// long double expm1l(long double x);
      expm1l,
      /// double fabs(double x);
      fabs,
      /// float fabsf(float x);
      fabsf,
      /// long double fabsl(long double x);
      fabsl,
      /// int fiprintf(FILE *stream, const char *format, ...);
      fiprintf,
      /// double floor(double x);
      floor,
      /// float floorf(float x);
      floorf,
      /// long double floorl(long double x);
      floorl,
      /// double fmod(double x, double y);
      fmod,
      /// float fmodf(float x, float y);
      fmodf,
      /// long double fmodl(long double x, long double y);
      fmodl,
      /// int fputc(int c, FILE *stream);
      fputc,
      /// int fputs(const char *s, FILE *stream);
      fputs,
      /// void free(void *ptr);
      free,
      /// size_t fwrite(const void *ptr, size_t size, size_t nitems,
      /// FILE *stream);
      fwrite,
      /// int iprintf(const char *format, ...);
      iprintf,
      /// double log(double x);
      log,
      /// double log10(double x);
      log10,
      /// float log10f(float x);
      log10f,
      /// long double log10l(long double x);
      log10l,
      /// double log1p(double x);
      log1p,
      /// float log1pf(float x);
      log1pf,
      /// long double log1pl(long double x);
      log1pl,
      /// double log2(double x);
      log2,
      /// float log2f(float x);
      log2f,
      /// double long double log2l(long double x);
      log2l,
      /// double logb(double x);
      logb,
      /// float logbf(float x);
      logbf,
      /// long double logbl(long double x);
      logbl,
      /// float logf(float x);
      logf,
      /// long double logl(long double x);
      logl,
      /// void *malloc(size_t size);
      malloc,
      /// void *memchr(const void *s, int c, size_t n);
      memchr,
      /// int memcmp(const void *s1, const void *s2, size_t n);
      memcmp,
      /// void *memcpy(void *s1, const void *s2, size_t n);
      memcpy,
      /// void *memmove(void *s1, const void *s2, size_t n);
      memmove,
      /// void *memset(void *b, int c, size_t len);
      memset,
      /// void memset_pattern16(void *b, const void *pattern16, size_t len);
      memset_pattern16,
      /// double nearbyint(double x);
      nearbyint,
      /// float nearbyintf(float x);
      nearbyintf,
      /// long double nearbyintl(long double x);
      nearbyintl,
      /// int posix_memalign(void **memptr, size_t alignment, size_t size);
      posix_memalign,
      /// double pow(double x, double y);
      pow,
      /// float powf(float x, float y);
      powf,
      /// long double powl(long double x, long double y);
      powl,
      /// int putchar(int c);
      putchar,
      /// int puts(const char *s);
      puts,
      /// void *realloc(void *ptr, size_t size);
      realloc,
      /// void *reallocf(void *ptr, size_t size);
      reallocf,
      /// double rint(double x);
      rint,
      /// float rintf(float x);
      rintf,
      /// long double rintl(long double x);
      rintl,
      /// double round(double x);
      round,
      /// float roundf(float x);
      roundf,
      /// long double roundl(long double x);
      roundl,
      /// double sin(double x);
      sin,
      /// float sinf(float x);
      sinf,
      /// double sinh(double x);
      sinh,
      /// float sinhf(float x);
      sinhf,
      /// long double sinhl(long double x);
      sinhl,
      /// long double sinl(long double x);
      sinl,
      /// int siprintf(char *str, const char *format, ...);
      siprintf,
      /// double sqrt(double x);
      sqrt,
      /// float sqrtf(float x);
      sqrtf,
      /// long double sqrtl(long double x);
      sqrtl,
      /// char *stpcpy(char *s1, const char *s2);
      stpcpy,
      /// char *strcat(char *s1, const char *s2);
      strcat,
      /// char *strchr(const char *s, int c);
      strchr,
      /// int strcmp(const char *s1, const char *s2);
      strcmp,
      /// char *strcpy(char *s1, const char *s2);
      strcpy,
      /// size_t strcspn(const char *s1, const char *s2);
      strcspn,
      /// char *strdup(const char *s1);
      strdup,
      /// size_t strlen(const char *s);
      strlen,
      /// char *strncat(char *s1, const char *s2, size_t n);
      strncat,
      /// int strncmp(const char *s1, const char *s2, size_t n);
      strncmp,
      /// char *strncpy(char *s1, const char *s2, size_t n);
      strncpy,
      /// char *strndup(const char *s1, size_t n);
      strndup,
      /// size_t strnlen(const char *s, size_t maxlen);
      strnlen,
      /// char *strpbrk(const char *s1, const char *s2);
      strpbrk,
      /// char *strrchr(const char *s, int c);
      strrchr,
      /// size_t strspn(const char *s1, const char *s2);
      strspn,
      /// char *strstr(const char *s1, const char *s2);
      strstr,
      /// double strtod(const char *nptr, char **endptr);
      strtod,
      /// float strtof(const char *nptr, char **endptr);
      strtof,
      /// long int strtol(const char *nptr, char **endptr, int base);
      strtol,
      /// long double strtold(const char *nptr, char **endptr);
      strtold,
      /// long long int strtoll(const char *nptr, char **endptr, int base);
      strtoll,
      /// unsigned long int strtoul(const char *nptr, char **endptr, int base);
      strtoul,
      /// unsigned long long int strtoull(const char *nptr, char **endptr,
      ///                                 int base);
      strtoull,
      /// double tan(double x);
      tan,
      /// float tanf(float x);
      tanf,
      /// double tanh(double x);
      tanh,
      /// float tanhf(float x);
      tanhf,
      /// long double tanhl(long double x);
      tanhl,
      /// long double tanl(long double x);
      tanl,
      /// double trunc(double x);
      trunc,
      /// float truncf(float x);
      truncf,
      /// long double truncl(long double x);
      truncl,
      /// void *valloc(size_t size);
      valloc,

      NumLibFuncs
    };
  }

/// TargetLibraryInfo - This immutable pass captures information about what
/// library functions are available for the current target, and allows a
/// frontend to disable optimizations through -fno-builtin etc.
class TargetLibraryInfo : public ImmutablePass {
  virtual void anchor();
  unsigned char AvailableArray[(LibFunc::NumLibFuncs+3)/4];
  llvm::DenseMap<unsigned, std::string> CustomNames;
  static const char* StandardNames[LibFunc::NumLibFuncs];

  enum AvailabilityState {
    StandardName = 3, // (memset to all ones)
    CustomName = 1,
    Unavailable = 0  // (memset to all zeros)
  };
  void setState(LibFunc::Func F, AvailabilityState State) {
    AvailableArray[F/4] &= ~(3 << 2*(F&3));
    AvailableArray[F/4] |= State << 2*(F&3);
  }
  AvailabilityState getState(LibFunc::Func F) const {
    return static_cast<AvailabilityState>((AvailableArray[F/4] >> 2*(F&3)) & 3);
  }

public:
  static char ID;
  TargetLibraryInfo();
  TargetLibraryInfo(const Triple &T);
  explicit TargetLibraryInfo(const TargetLibraryInfo &TLI);
  
  /// getLibFunc - Search for a particular function name.  If it is one of the
  /// known library functions, return true and set F to the corresponding value.
  bool getLibFunc(StringRef funcName, LibFunc::Func &F) const;

  /// has - This function is used by optimizations that want to match on or form
  /// a given library function.
  bool has(LibFunc::Func F) const {
    return getState(F) != Unavailable;
  }

  /// hasOptimizedCodeGen - Return true if the function is both available as
  /// a builtin and a candidate for optimized code generation.
  bool hasOptimizedCodeGen(LibFunc::Func F) const {
    if (getState(F) == Unavailable)
      return false;
    switch (F) {
    default: break;
    case LibFunc::copysign:  case LibFunc::copysignf:  case LibFunc::copysignl:
    case LibFunc::fabs:      case LibFunc::fabsf:      case LibFunc::fabsl:
    case LibFunc::sin:       case LibFunc::sinf:       case LibFunc::sinl:
    case LibFunc::cos:       case LibFunc::cosf:       case LibFunc::cosl:
    case LibFunc::sqrt:      case LibFunc::sqrtf:      case LibFunc::sqrtl:
    case LibFunc::floor:     case LibFunc::floorf:     case LibFunc::floorl:
    case LibFunc::nearbyint: case LibFunc::nearbyintf: case LibFunc::nearbyintl:
    case LibFunc::ceil:      case LibFunc::ceilf:      case LibFunc::ceill:
    case LibFunc::rint:      case LibFunc::rintf:      case LibFunc::rintl:
    case LibFunc::trunc:     case LibFunc::truncf:     case LibFunc::truncl:
    case LibFunc::log2:      case LibFunc::log2f:      case LibFunc::log2l:
    case LibFunc::exp2:      case LibFunc::exp2f:      case LibFunc::exp2l:
    case LibFunc::memcmp:
      return true;
    }
    return false;
  }

  StringRef getName(LibFunc::Func F) const {
    AvailabilityState State = getState(F);
    if (State == Unavailable)
      return StringRef();
    if (State == StandardName)
      return StandardNames[F];
    assert(State == CustomName);
    return CustomNames.find(F)->second;
  }

  /// setUnavailable - this can be used by whatever sets up TargetLibraryInfo to
  /// ban use of specific library functions.
  void setUnavailable(LibFunc::Func F) {
    setState(F, Unavailable);
  }

  void setAvailable(LibFunc::Func F) {
    setState(F, StandardName);
  }

  void setAvailableWithName(LibFunc::Func F, StringRef Name) {
    if (StandardNames[F] != Name) {
      setState(F, CustomName);
      CustomNames[F] = Name;
      assert(CustomNames.find(F) != CustomNames.end());
    } else {
      setState(F, StandardName);
    }
  }

  /// disableAllFunctions - This disables all builtins, which is used for
  /// options like -fno-builtin.
  void disableAllFunctions();
};

} // end namespace llvm

#endif
