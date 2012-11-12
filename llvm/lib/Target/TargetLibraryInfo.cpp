//===-- TargetLibraryInfo.cpp - Runtime library information ----------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TargetLibraryInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;

// Register the default implementation.
INITIALIZE_PASS(TargetLibraryInfo, "targetlibinfo",
                "Target Library Information", false, true)
char TargetLibraryInfo::ID = 0;

void TargetLibraryInfo::anchor() { }

const char* TargetLibraryInfo::StandardNames[LibFunc::NumLibFuncs] =
  {
    "_ZdaPv",
    "_ZdlPv",
    "_Znaj",
    "_ZnajRKSt9nothrow_t",
    "_Znam",
    "_ZnamRKSt9nothrow_t",
    "_Znwj",
    "_ZnwjRKSt9nothrow_t",
    "_Znwm",
    "_ZnwmRKSt9nothrow_t",
    "__cxa_atexit",
    "__cxa_guard_abort",
    "__cxa_guard_acquire",
    "__cxa_guard_release",
    "__memcpy_chk",
    "acos",
    "acosf",
    "acosh",
    "acoshf",
    "acoshl",
    "acosl",
    "asin",
    "asinf",
    "asinh",
    "asinhf",
    "asinhl",
    "asinl",
    "atan",
    "atan2",
    "atan2f",
    "atan2l",
    "atanf",
    "atanh",
    "atanhf",
    "atanhl",
    "atanl",
    "calloc",
    "cbrt",
    "cbrtf",
    "cbrtl",
    "ceil",
    "ceilf",
    "ceill",
    "copysign",
    "copysignf",
    "copysignl",
    "cos",
    "cosf",
    "cosh",
    "coshf",
    "coshl",
    "cosl",
    "exp",
    "exp10",
    "exp10f",
    "exp10l",
    "exp2",
    "exp2f",
    "exp2l",
    "expf",
    "expl",
    "expm1",
    "expm1f",
    "expm1l",
    "fabs",
    "fabsf",
    "fabsl",
    "fiprintf",
    "floor",
    "floorf",
    "floorl",
    "fmod",
    "fmodf",
    "fmodl",
    "fputc",
    "fputs",
    "free",
    "fwrite",
    "iprintf",
    "log",
    "log10",
    "log10f",
    "log10l",
    "log1p",
    "log1pf",
    "log1pl",
    "log2",
    "log2f",
    "log2l",
    "logb",
    "logbf",
    "logbl",
    "logf",
    "logl",
    "malloc",
    "memchr",
    "memcmp",
    "memcpy",
    "memmove",
    "memset",
    "memset_pattern16",
    "nearbyint",
    "nearbyintf",
    "nearbyintl",
    "posix_memalign",
    "pow",
    "powf",
    "powl",
    "putchar",
    "puts",
    "realloc",
    "reallocf",
    "rint",
    "rintf",
    "rintl",
    "round",
    "roundf",
    "roundl",
    "sin",
    "sinf",
    "sinh",
    "sinhf",
    "sinhl",
    "sinl",
    "siprintf",
    "sqrt",
    "sqrtf",
    "sqrtl",
    "stpcpy",
    "strcat",
    "strchr",
    "strcmp",
    "strcpy",
    "strcspn",
    "strdup",
    "strlen",
    "strncat",
    "strncmp",
    "strncpy",
    "strndup",
    "strnlen",
    "strpbrk",
    "strrchr",
    "strspn",
    "strstr",
    "strtod",
    "strtof",
    "strtol",
    "strtold",
    "strtoll",
    "strtoul",
    "strtoull",
    "tan",
    "tanf",
    "tanh",
    "tanhf",
    "tanhl",
    "tanl",
    "trunc",
    "truncf",
    "truncl",
    "valloc"
  };

/// initialize - Initialize the set of available library functions based on the
/// specified target triple.  This should be carefully written so that a missing
/// target triple gets a sane set of defaults.
static void initialize(TargetLibraryInfo &TLI, const Triple &T,
                       const char **StandardNames) {
  initializeTargetLibraryInfoPass(*PassRegistry::getPassRegistry());

#ifndef NDEBUG
  // Verify that the StandardNames array is in alphabetical order.
  for (unsigned F = 1; F < LibFunc::NumLibFuncs; ++F) {
    if (strcmp(StandardNames[F-1], StandardNames[F]) >= 0)
      llvm_unreachable("TargetLibraryInfo function names must be sorted");
  }
#endif // !NDEBUG
  
  // memset_pattern16 is only available on iOS 3.0 and Mac OS/X 10.5 and later.
  if (T.isMacOSX()) {
    if (T.isMacOSXVersionLT(10, 5))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else if (T.getOS() == Triple::IOS) {
    if (T.isOSVersionLT(3, 0))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else {
    TLI.setUnavailable(LibFunc::memset_pattern16);
  }

  if (T.isMacOSX() && T.getArch() == Triple::x86 &&
      !T.isMacOSXVersionLT(10, 7)) {
    // x86-32 OSX has a scheme where fwrite and fputs (and some other functions
    // we don't care about) have two versions; on recent OSX, the one we want
    // has a $UNIX2003 suffix. The two implementations are identical except
    // for the return value in some edge cases.  However, we don't want to
    // generate code that depends on the old symbols.
    TLI.setAvailableWithName(LibFunc::fwrite, "fwrite$UNIX2003");
    TLI.setAvailableWithName(LibFunc::fputs, "fputs$UNIX2003");
  }

  // iprintf and friends are only available on XCore and TCE.
  if (T.getArch() != Triple::xcore && T.getArch() != Triple::tce) {
    TLI.setUnavailable(LibFunc::iprintf);
    TLI.setUnavailable(LibFunc::siprintf);
    TLI.setUnavailable(LibFunc::fiprintf);
  }

  if (T.getOS() == Triple::Win32) {
    // Win32 does not support long double
    TLI.setUnavailable(LibFunc::acosl);
    TLI.setUnavailable(LibFunc::asinl);
    TLI.setUnavailable(LibFunc::atanl);
    TLI.setUnavailable(LibFunc::atan2l);
    TLI.setUnavailable(LibFunc::ceill);
    TLI.setUnavailable(LibFunc::copysignl);
    TLI.setUnavailable(LibFunc::cosl);
    TLI.setUnavailable(LibFunc::coshl);
    TLI.setUnavailable(LibFunc::expl);
    TLI.setUnavailable(LibFunc::fabsf); // Win32 and Win64 both lack fabsf
    TLI.setUnavailable(LibFunc::fabsl);
    TLI.setUnavailable(LibFunc::floorl);
    TLI.setUnavailable(LibFunc::fmodl);
    TLI.setUnavailable(LibFunc::logl);
    TLI.setUnavailable(LibFunc::powl);
    TLI.setUnavailable(LibFunc::sinl);
    TLI.setUnavailable(LibFunc::sinhl);
    TLI.setUnavailable(LibFunc::sqrtl);
    TLI.setUnavailable(LibFunc::tanl);
    TLI.setUnavailable(LibFunc::tanhl);

    // Win32 only has C89 math
    TLI.setUnavailable(LibFunc::acosh);
    TLI.setUnavailable(LibFunc::acoshf);
    TLI.setUnavailable(LibFunc::acoshl);
    TLI.setUnavailable(LibFunc::asinh);
    TLI.setUnavailable(LibFunc::asinhf);
    TLI.setUnavailable(LibFunc::asinhl);
    TLI.setUnavailable(LibFunc::atanh);
    TLI.setUnavailable(LibFunc::atanhf);
    TLI.setUnavailable(LibFunc::atanhl);
    TLI.setUnavailable(LibFunc::cbrt);
    TLI.setUnavailable(LibFunc::cbrtf);
    TLI.setUnavailable(LibFunc::cbrtl);
    TLI.setUnavailable(LibFunc::exp10);
    TLI.setUnavailable(LibFunc::exp10f);
    TLI.setUnavailable(LibFunc::exp10l);
    TLI.setUnavailable(LibFunc::exp2);
    TLI.setUnavailable(LibFunc::exp2f);
    TLI.setUnavailable(LibFunc::exp2l);
    TLI.setUnavailable(LibFunc::expm1);
    TLI.setUnavailable(LibFunc::expm1f);
    TLI.setUnavailable(LibFunc::expm1l);
    TLI.setUnavailable(LibFunc::log2);
    TLI.setUnavailable(LibFunc::log2f);
    TLI.setUnavailable(LibFunc::log2l);
    TLI.setUnavailable(LibFunc::log1p);
    TLI.setUnavailable(LibFunc::log1pf);
    TLI.setUnavailable(LibFunc::log1pl);
    TLI.setUnavailable(LibFunc::logb);
    TLI.setUnavailable(LibFunc::logbf);
    TLI.setUnavailable(LibFunc::logbl);
    TLI.setUnavailable(LibFunc::nearbyint);
    TLI.setUnavailable(LibFunc::nearbyintf);
    TLI.setUnavailable(LibFunc::nearbyintl);
    TLI.setUnavailable(LibFunc::rint);
    TLI.setUnavailable(LibFunc::rintf);
    TLI.setUnavailable(LibFunc::rintl);
    TLI.setUnavailable(LibFunc::round);
    TLI.setUnavailable(LibFunc::roundf);
    TLI.setUnavailable(LibFunc::roundl);
    TLI.setUnavailable(LibFunc::trunc);
    TLI.setUnavailable(LibFunc::truncf);
    TLI.setUnavailable(LibFunc::truncl);

    // Win32 provides some C99 math with mangled names
    TLI.setAvailableWithName(LibFunc::copysign, "_copysign");

    if (T.getArch() == Triple::x86) {
      // Win32 on x86 implements single-precision math functions as macros
      TLI.setUnavailable(LibFunc::acosf);
      TLI.setUnavailable(LibFunc::asinf);
      TLI.setUnavailable(LibFunc::atanf);
      TLI.setUnavailable(LibFunc::atan2f);
      TLI.setUnavailable(LibFunc::ceilf);
      TLI.setUnavailable(LibFunc::copysignf);
      TLI.setUnavailable(LibFunc::cosf);
      TLI.setUnavailable(LibFunc::coshf);
      TLI.setUnavailable(LibFunc::expf);
      TLI.setUnavailable(LibFunc::floorf);
      TLI.setUnavailable(LibFunc::fmodf);
      TLI.setUnavailable(LibFunc::logf);
      TLI.setUnavailable(LibFunc::powf);
      TLI.setUnavailable(LibFunc::sinf);
      TLI.setUnavailable(LibFunc::sinhf);
      TLI.setUnavailable(LibFunc::sqrtf);
      TLI.setUnavailable(LibFunc::tanf);
      TLI.setUnavailable(LibFunc::tanhf);
    }

    // Win32 does *not* provide stpcpy.  It is provided on POSIX systems:
    // http://pubs.opengroup.org/onlinepubs/9699919799/functions/stpcpy.html
    TLI.setUnavailable(LibFunc::stpcpy);
  }
}


TargetLibraryInfo::TargetLibraryInfo() : ImmutablePass(ID) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));

  initialize(*this, Triple(), StandardNames);
}

TargetLibraryInfo::TargetLibraryInfo(const Triple &T) : ImmutablePass(ID) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));
  
  initialize(*this, T, StandardNames);
}

TargetLibraryInfo::TargetLibraryInfo(const TargetLibraryInfo &TLI)
  : ImmutablePass(ID) {
  memcpy(AvailableArray, TLI.AvailableArray, sizeof(AvailableArray));
  CustomNames = TLI.CustomNames;
}

bool TargetLibraryInfo::getLibFunc(StringRef funcName,
                                   LibFunc::Func &F) const {
  const char **Start = &StandardNames[0];
  const char **End = &StandardNames[LibFunc::NumLibFuncs];
  const char **I = std::lower_bound(Start, End, funcName);
  if (I != End && *I == funcName) {
    F = (LibFunc::Func)(I - Start);
    return true;
  }
  return false;
}

/// disableAllFunctions - This disables all builtins, which is used for options
/// like -fno-builtin.
void TargetLibraryInfo::disableAllFunctions() {
  memset(AvailableArray, 0, sizeof(AvailableArray));
}
