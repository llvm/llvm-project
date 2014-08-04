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
    "_IO_getc",
    "_IO_putc",
    "_ZdaPv",
    "_ZdaPvRKSt9nothrow_t",
    "_ZdlPv",
    "_ZdlPvRKSt9nothrow_t",
    "_Znaj",
    "_ZnajRKSt9nothrow_t",
    "_Znam",
    "_ZnamRKSt9nothrow_t",
    "_Znwj",
    "_ZnwjRKSt9nothrow_t",
    "_Znwm",
    "_ZnwmRKSt9nothrow_t",
    "__cospi",
    "__cospif",
    "__cxa_atexit",
    "__cxa_guard_abort",
    "__cxa_guard_acquire",
    "__cxa_guard_release",
    "__isoc99_scanf",
    "__isoc99_sscanf",
    "__memcpy_chk",
    "__sincospi_stret",
    "__sincospif_stret",
    "__sinpi",
    "__sinpif",
    "__sqrt_finite",
    "__sqrtf_finite",
    "__sqrtl_finite",
    "__strdup",
    "__strndup",
    "__strtok_r",
    "abs",
    "access",
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
    "atof",
    "atoi",
    "atol",
    "atoll",
    "bcmp",
    "bcopy",
    "bzero",
    "calloc",
    "cbrt",
    "cbrtf",
    "cbrtl",
    "ceil",
    "ceilf",
    "ceill",
    "chmod",
    "chown",
    "clearerr",
    "closedir",
    "copysign",
    "copysignf",
    "copysignl",
    "cos",
    "cosf",
    "cosh",
    "coshf",
    "coshl",
    "cosl",
    "ctermid",
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
    "fclose",
    "fdopen",
    "feof",
    "ferror",
    "fflush",
    "ffs",
    "ffsl",
    "ffsll",
    "fgetc",
    "fgetpos",
    "fgets",
    "fileno",
    "fiprintf",
    "flockfile",
    "floor",
    "floorf",
    "floorl",
    "fmax",
    "fmaxf",
    "fmaxl",
    "fmin",
    "fminf",
    "fminl",
    "fmod",
    "fmodf",
    "fmodl",
    "fopen",
    "fopen64",
    "fprintf",
    "fputc",
    "fputs",
    "fread",
    "free",
    "frexp",
    "frexpf",
    "frexpl",
    "fscanf",
    "fseek",
    "fseeko",
    "fseeko64",
    "fsetpos",
    "fstat",
    "fstat64",
    "fstatvfs",
    "fstatvfs64",
    "ftell",
    "ftello",
    "ftello64",
    "ftrylockfile",
    "funlockfile",
    "fwrite",
    "getc",
    "getc_unlocked",
    "getchar",
    "getenv",
    "getitimer",
    "getlogin_r",
    "getpwnam",
    "gets",
    "gettimeofday",
    "htonl",
    "htons",
    "iprintf",
    "isascii",
    "isdigit",
    "labs",
    "lchown",
    "ldexp",
    "ldexpf",
    "ldexpl",
    "llabs",
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
    "lstat",
    "lstat64",
    "malloc",
    "memalign",
    "memccpy",
    "memchr",
    "memcmp",
    "memcpy",
    "memmove",
    "memrchr",
    "memset",
    "memset_pattern16",
    "mkdir",
    "mktime",
    "modf",
    "modff",
    "modfl",
    "nearbyint",
    "nearbyintf",
    "nearbyintl",
    "ntohl",
    "ntohs",
    "open",
    "open64",
    "opendir",
    "pclose",
    "perror",
    "popen",
    "posix_memalign",
    "pow",
    "powf",
    "powl",
    "pread",
    "printf",
    "putc",
    "putchar",
    "puts",
    "pwrite",
    "qsort",
    "read",
    "readlink",
    "realloc",
    "reallocf",
    "realpath",
    "remove",
    "rename",
    "rewind",
    "rint",
    "rintf",
    "rintl",
    "rmdir",
    "round",
    "roundf",
    "roundl",
    "scanf",
    "setbuf",
    "setitimer",
    "setvbuf",
    "sin",
    "sinf",
    "sinh",
    "sinhf",
    "sinhl",
    "sinl",
    "siprintf",
    "snprintf",
    "sprintf",
    "sqrt",
    "sqrtf",
    "sqrtl",
    "sscanf",
    "stat",
    "stat64",
    "statvfs",
    "statvfs64",
    "stpcpy",
    "stpncpy",
    "strcasecmp",
    "strcat",
    "strchr",
    "strcmp",
    "strcoll",
    "strcpy",
    "strcspn",
    "strdup",
    "strlen",
    "strncasecmp",
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
    "strtok",
    "strtok_r",
    "strtol",
    "strtold",
    "strtoll",
    "strtoul",
    "strtoull",
    "strxfrm",
    "system",
    "tan",
    "tanf",
    "tanh",
    "tanhf",
    "tanhl",
    "tanl",
    "times",
    "tmpfile",
    "tmpfile64",
    "toascii",
    "trunc",
    "truncf",
    "truncl",
    "uname",
    "ungetc",
    "unlink",
    "unsetenv",
    "utime",
    "utimes",
    "valloc",
    "vfprintf",
    "vfscanf",
    "vprintf",
    "vscanf",
    "vsnprintf",
    "vsprintf",
    "vsscanf",
    "write"
  };

static bool hasSinCosPiStret(const Triple &T) {
  // Only Darwin variants have _stret versions of combined trig functions.
  if (!T.isMacOSX() && T.getOS() != Triple::IOS)
    return false;

  // The ABI is rather complicated on x86, so don't do anything special there.
  if (T.getArch() == Triple::x86)
    return false;

  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 9))
    return false;

  if (T.getOS() == Triple::IOS && T.isOSVersionLT(7, 0))
    return false;

  return true;
}

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

  // There are no library implementations of mempcy and memset for r600 and
  // these can be difficult to lower in the backend.
  if (T.getArch() == Triple::r600) {
    TLI.setUnavailable(LibFunc::memcpy);
    TLI.setUnavailable(LibFunc::memset);
    TLI.setUnavailable(LibFunc::memset_pattern16);
    return;
  }

  // memset_pattern16 is only available on iOS 3.0 and Mac OS X 10.5 and later.
  if (T.isMacOSX()) {
    if (T.isMacOSXVersionLT(10, 5))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else if (T.isiOS()) {
    if (T.isOSVersionLT(3, 0))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else {
    TLI.setUnavailable(LibFunc::memset_pattern16);
  }

  if (!hasSinCosPiStret(T)) {
    TLI.setUnavailable(LibFunc::sinpi);
    TLI.setUnavailable(LibFunc::sinpif);
    TLI.setUnavailable(LibFunc::cospi);
    TLI.setUnavailable(LibFunc::cospif);
    TLI.setUnavailable(LibFunc::sincospi_stret);
    TLI.setUnavailable(LibFunc::sincospif_stret);
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

  if (T.isOSWindows() && !T.isOSCygMing()) {
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
    TLI.setUnavailable(LibFunc::fmaxl);
    TLI.setUnavailable(LibFunc::fminl);
    TLI.setUnavailable(LibFunc::fmodl);
    TLI.setUnavailable(LibFunc::frexpl);
    TLI.setUnavailable(LibFunc::ldexpf);
    TLI.setUnavailable(LibFunc::ldexpl);
    TLI.setUnavailable(LibFunc::logl);
    TLI.setUnavailable(LibFunc::modfl);
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
      TLI.setUnavailable(LibFunc::fminf);
      TLI.setUnavailable(LibFunc::fmaxf);
      TLI.setUnavailable(LibFunc::fmodf);
      TLI.setUnavailable(LibFunc::logf);
      TLI.setUnavailable(LibFunc::powf);
      TLI.setUnavailable(LibFunc::sinf);
      TLI.setUnavailable(LibFunc::sinhf);
      TLI.setUnavailable(LibFunc::sqrtf);
      TLI.setUnavailable(LibFunc::tanf);
      TLI.setUnavailable(LibFunc::tanhf);
    }

    // Win32 does *not* provide provide these functions, but they are
    // generally available on POSIX-compliant systems:
    TLI.setUnavailable(LibFunc::access);
    TLI.setUnavailable(LibFunc::bcmp);
    TLI.setUnavailable(LibFunc::bcopy);
    TLI.setUnavailable(LibFunc::bzero);
    TLI.setUnavailable(LibFunc::chmod);
    TLI.setUnavailable(LibFunc::chown);
    TLI.setUnavailable(LibFunc::closedir);
    TLI.setUnavailable(LibFunc::ctermid);
    TLI.setUnavailable(LibFunc::fdopen);
    TLI.setUnavailable(LibFunc::ffs);
    TLI.setUnavailable(LibFunc::fileno);
    TLI.setUnavailable(LibFunc::flockfile);
    TLI.setUnavailable(LibFunc::fseeko);
    TLI.setUnavailable(LibFunc::fstat);
    TLI.setUnavailable(LibFunc::fstatvfs);
    TLI.setUnavailable(LibFunc::ftello);
    TLI.setUnavailable(LibFunc::ftrylockfile);
    TLI.setUnavailable(LibFunc::funlockfile);
    TLI.setUnavailable(LibFunc::getc_unlocked);
    TLI.setUnavailable(LibFunc::getitimer);
    TLI.setUnavailable(LibFunc::getlogin_r);
    TLI.setUnavailable(LibFunc::getpwnam);
    TLI.setUnavailable(LibFunc::gettimeofday);
    TLI.setUnavailable(LibFunc::htonl);
    TLI.setUnavailable(LibFunc::htons);
    TLI.setUnavailable(LibFunc::lchown);
    TLI.setUnavailable(LibFunc::lstat);
    TLI.setUnavailable(LibFunc::memccpy);
    TLI.setUnavailable(LibFunc::mkdir);
    TLI.setUnavailable(LibFunc::ntohl);
    TLI.setUnavailable(LibFunc::ntohs);
    TLI.setUnavailable(LibFunc::open);
    TLI.setUnavailable(LibFunc::opendir);
    TLI.setUnavailable(LibFunc::pclose);
    TLI.setUnavailable(LibFunc::popen);
    TLI.setUnavailable(LibFunc::pread);
    TLI.setUnavailable(LibFunc::pwrite);
    TLI.setUnavailable(LibFunc::read);
    TLI.setUnavailable(LibFunc::readlink);
    TLI.setUnavailable(LibFunc::realpath);
    TLI.setUnavailable(LibFunc::rmdir);
    TLI.setUnavailable(LibFunc::setitimer);
    TLI.setUnavailable(LibFunc::stat);
    TLI.setUnavailable(LibFunc::statvfs);
    TLI.setUnavailable(LibFunc::stpcpy);
    TLI.setUnavailable(LibFunc::stpncpy);
    TLI.setUnavailable(LibFunc::strcasecmp);
    TLI.setUnavailable(LibFunc::strncasecmp);
    TLI.setUnavailable(LibFunc::times);
    TLI.setUnavailable(LibFunc::uname);
    TLI.setUnavailable(LibFunc::unlink);
    TLI.setUnavailable(LibFunc::unsetenv);
    TLI.setUnavailable(LibFunc::utime);
    TLI.setUnavailable(LibFunc::utimes);
    TLI.setUnavailable(LibFunc::write);

    // Win32 does *not* provide provide these functions, but they are
    // specified by C99:
    TLI.setUnavailable(LibFunc::atoll);
    TLI.setUnavailable(LibFunc::frexpf);
    TLI.setUnavailable(LibFunc::llabs);
  }

  switch (T.getOS()) {
  case Triple::MacOSX:
    // exp10 and exp10f are not available on OS X until 10.9 and iOS until 7.0
    // and their names are __exp10 and __exp10f. exp10l is not available on
    // OS X or iOS.
    TLI.setUnavailable(LibFunc::exp10l);
    if (T.isMacOSXVersionLT(10, 9)) {
      TLI.setUnavailable(LibFunc::exp10);
      TLI.setUnavailable(LibFunc::exp10f);
    } else {
      TLI.setAvailableWithName(LibFunc::exp10, "__exp10");
      TLI.setAvailableWithName(LibFunc::exp10f, "__exp10f");
    }
    break;
  case Triple::IOS:
    TLI.setUnavailable(LibFunc::exp10l);
    if (T.isOSVersionLT(7, 0)) {
      TLI.setUnavailable(LibFunc::exp10);
      TLI.setUnavailable(LibFunc::exp10f);
    } else {
      TLI.setAvailableWithName(LibFunc::exp10, "__exp10");
      TLI.setAvailableWithName(LibFunc::exp10f, "__exp10f");
    }
    break;
  case Triple::Linux:
    // exp10, exp10f, exp10l is available on Linux (GLIBC) but are extremely
    // buggy prior to glibc version 2.18. Until this version is widely deployed
    // or we have a reasonable detection strategy, we cannot use exp10 reliably
    // on Linux.
    //
    // Fall through to disable all of them.
  default:
    TLI.setUnavailable(LibFunc::exp10);
    TLI.setUnavailable(LibFunc::exp10f);
    TLI.setUnavailable(LibFunc::exp10l);
  }

  // ffsl is available on at least Darwin, Mac OS X, iOS, FreeBSD, and
  // Linux (GLIBC):
  // http://developer.apple.com/library/mac/#documentation/Darwin/Reference/ManPages/man3/ffsl.3.html
  // http://svn.freebsd.org/base/user/eri/pf45/head/lib/libc/string/ffsl.c
  // http://www.gnu.org/software/gnulib/manual/html_node/ffsl.html
  switch (T.getOS()) {
  case Triple::Darwin:
  case Triple::MacOSX:
  case Triple::IOS:
  case Triple::FreeBSD:
  case Triple::Linux:
    break;
  default:
    TLI.setUnavailable(LibFunc::ffsl);
  }

  // ffsll is available on at least FreeBSD and Linux (GLIBC):
  // http://svn.freebsd.org/base/user/eri/pf45/head/lib/libc/string/ffsll.c
  // http://www.gnu.org/software/gnulib/manual/html_node/ffsll.html
  switch (T.getOS()) {
  case Triple::FreeBSD:
  case Triple::Linux:
    break;
  default:
    TLI.setUnavailable(LibFunc::ffsll);
  }

  // The following functions are available on at least Linux:
  if (!T.isOSLinux()) {
    TLI.setUnavailable(LibFunc::dunder_strdup);
    TLI.setUnavailable(LibFunc::dunder_strtok_r);
    TLI.setUnavailable(LibFunc::dunder_isoc99_scanf);
    TLI.setUnavailable(LibFunc::dunder_isoc99_sscanf);
    TLI.setUnavailable(LibFunc::under_IO_getc);
    TLI.setUnavailable(LibFunc::under_IO_putc);
    TLI.setUnavailable(LibFunc::memalign);
    TLI.setUnavailable(LibFunc::fopen64);
    TLI.setUnavailable(LibFunc::fseeko64);
    TLI.setUnavailable(LibFunc::fstat64);
    TLI.setUnavailable(LibFunc::fstatvfs64);
    TLI.setUnavailable(LibFunc::ftello64);
    TLI.setUnavailable(LibFunc::lstat64);
    TLI.setUnavailable(LibFunc::open64);
    TLI.setUnavailable(LibFunc::stat64);
    TLI.setUnavailable(LibFunc::statvfs64);
    TLI.setUnavailable(LibFunc::tmpfile64);
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

namespace {
struct StringComparator {
  /// Compare two strings and return true if LHS is lexicographically less than
  /// RHS. Requires that RHS doesn't contain any zero bytes.
  bool operator()(const char *LHS, StringRef RHS) const {
    // Compare prefixes with strncmp. If prefixes match we know that LHS is
    // greater or equal to RHS as RHS can't contain any '\0'.
    return std::strncmp(LHS, RHS.data(), RHS.size()) < 0;
  }

  // Provided for compatibility with MSVC's debug mode.
  bool operator()(StringRef LHS, const char *RHS) const { return LHS < RHS; }
  bool operator()(StringRef LHS, StringRef RHS) const { return LHS < RHS; }
  bool operator()(const char *LHS, const char *RHS) const {
    return std::strcmp(LHS, RHS) < 0;
  }
};
}

bool TargetLibraryInfo::getLibFunc(StringRef funcName,
                                   LibFunc::Func &F) const {
  const char **Start = &StandardNames[0];
  const char **End = &StandardNames[LibFunc::NumLibFuncs];

  // Filter out empty names and names containing null bytes, those can't be in
  // our table.
  if (funcName.empty() || funcName.find('\0') != StringRef::npos)
    return false;

  // Check for \01 prefix that is used to mangle __asm declarations and
  // strip it if present.
  if (funcName.front() == '\01')
    funcName = funcName.substr(1);
  const char **I = std::lower_bound(Start, End, funcName, StringComparator());
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
