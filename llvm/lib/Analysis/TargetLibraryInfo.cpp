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

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<TargetLibraryInfoImpl::VectorLibrary> ClVectorLibrary(
    "vector-library", cl::Hidden, cl::desc("Vector functions library"),
    cl::init(TargetLibraryInfoImpl::NoLibrary),
    cl::values(clEnumValN(TargetLibraryInfoImpl::NoLibrary, "none",
                          "No vector functions library"),
               clEnumValN(TargetLibraryInfoImpl::Accelerate, "Accelerate",
                          "Accelerate framework"),
               clEnumValN(TargetLibraryInfoImpl::SVML, "SVML",
                          "Intel SVML library"),
               clEnumValEnd));

const char *const TargetLibraryInfoImpl::StandardNames[LibFunc::NumLibFuncs] = {
#define TLI_DEFINE_STRING
#include "llvm/Analysis/TargetLibraryInfo.def"
};

static bool hasSinCosPiStret(const Triple &T) {
  // Only Darwin variants have _stret versions of combined trig functions.
  if (!T.isOSDarwin())
    return false;

  // The ABI is rather complicated on x86, so don't do anything special there.
  if (T.getArch() == Triple::x86)
    return false;

  if (T.isMacOSX() && T.isMacOSXVersionLT(10, 9))
    return false;

  if (T.isiOS() && T.isOSVersionLT(7, 0))
    return false;

  return true;
}

/// initialize - Initialize the set of available library functions based on the
/// specified target triple.  This should be carefully written so that a missing
/// target triple gets a sane set of defaults.
static void initialize(TargetLibraryInfoImpl &TLI, const Triple &T,
                       ArrayRef<const char *> StandardNames) {
  // Verify that the StandardNames array is in alphabetical order.
  assert(std::is_sorted(StandardNames.begin(), StandardNames.end(),
                        [](const char *LHS, const char *RHS) {
                          return strcmp(LHS, RHS) < 0;
                        }) &&
         "TargetLibraryInfoImpl function names must be sorted");

  if (T.getArch() == Triple::r600 ||
      T.getArch() == Triple::amdgcn) {
    TLI.setUnavailable(LibFunc::ldexp);
    TLI.setUnavailable(LibFunc::ldexpf);
    TLI.setUnavailable(LibFunc::ldexpl);
    TLI.setUnavailable(LibFunc::exp10);
    TLI.setUnavailable(LibFunc::exp10f);
    TLI.setUnavailable(LibFunc::exp10l);
    TLI.setUnavailable(LibFunc::log10);
    TLI.setUnavailable(LibFunc::log10f);
    TLI.setUnavailable(LibFunc::log10l);
  }

  // There are no library implementations of mempcy and memset for AMD gpus and
  // these can be difficult to lower in the backend.
  if (T.getArch() == Triple::r600 ||
      T.getArch() == Triple::amdgcn) {
    TLI.setUnavailable(LibFunc::memcpy);
    TLI.setUnavailable(LibFunc::memset);
    TLI.setUnavailable(LibFunc::memset_pattern16);
    return;
  }

  // memset_pattern16 is only available on iOS 3.0 and Mac OS X 10.5 and later.
  // All versions of watchOS support it.
  if (T.isMacOSX()) {
    if (T.isMacOSXVersionLT(10, 5))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else if (T.isiOS()) {
    if (T.isOSVersionLT(3, 0))
      TLI.setUnavailable(LibFunc::memset_pattern16);
  } else if (!T.isWatchOS()) {
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
      TLI.setUnavailable(LibFunc::log10f);
      TLI.setUnavailable(LibFunc::modff);
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
  case Triple::TvOS:
  case Triple::WatchOS:
    TLI.setUnavailable(LibFunc::exp10l);
    if (!T.isWatchOS() && (T.isOSVersionLT(7, 0) ||
                           (T.isOSVersionLT(9, 0) &&
                            (T.getArch() == Triple::x86 ||
                             T.getArch() == Triple::x86_64)))) {
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
    LLVM_FALLTHROUGH;
  default:
    TLI.setUnavailable(LibFunc::exp10);
    TLI.setUnavailable(LibFunc::exp10f);
    TLI.setUnavailable(LibFunc::exp10l);
  }

  // ffsl is available on at least Darwin, Mac OS X, iOS, FreeBSD, and
  // Linux (GLIBC):
  // http://developer.apple.com/library/mac/#documentation/Darwin/Reference/ManPages/man3/ffsl.3.html
  // http://svn.freebsd.org/base/head/lib/libc/string/ffsl.c
  // http://www.gnu.org/software/gnulib/manual/html_node/ffsl.html
  switch (T.getOS()) {
  case Triple::Darwin:
  case Triple::MacOSX:
  case Triple::IOS:
  case Triple::TvOS:
  case Triple::WatchOS:
  case Triple::FreeBSD:
  case Triple::Linux:
    break;
  default:
    TLI.setUnavailable(LibFunc::ffsl);
  }

  // ffsll is available on at least FreeBSD and Linux (GLIBC):
  // http://svn.freebsd.org/base/head/lib/libc/string/ffsll.c
  // http://www.gnu.org/software/gnulib/manual/html_node/ffsll.html
  switch (T.getOS()) {
  case Triple::Darwin:
  case Triple::MacOSX:
  case Triple::IOS:
  case Triple::TvOS:
  case Triple::WatchOS:
  case Triple::FreeBSD:
  case Triple::Linux:
    break;
  default:
    TLI.setUnavailable(LibFunc::ffsll);
  }

  // The following functions are available on at least FreeBSD:
  // http://svn.freebsd.org/base/head/lib/libc/string/fls.c
  // http://svn.freebsd.org/base/head/lib/libc/string/flsl.c
  // http://svn.freebsd.org/base/head/lib/libc/string/flsll.c
  if (!T.isOSFreeBSD()) {
    TLI.setUnavailable(LibFunc::fls);
    TLI.setUnavailable(LibFunc::flsl);
    TLI.setUnavailable(LibFunc::flsll);
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

  // As currently implemented in clang, NVPTX code has no standard library to
  // speak of.  Headers provide a standard-ish library implementation, but many
  // of the signatures are wrong -- for example, many libm functions are not
  // extern "C".
  //
  // libdevice, an IR library provided by nvidia, is linked in by the front-end,
  // but only used functions are provided to llvm.  Moreover, most of the
  // functions in libdevice don't map precisely to standard library functions.
  //
  // FIXME: Having no standard library prevents e.g. many fastmath
  // optimizations, so this situation should be fixed.
  if (T.isNVPTX()) {
    TLI.disableAllFunctions();
    TLI.setAvailable(LibFunc::nvvm_reflect);
  } else {
    TLI.setUnavailable(LibFunc::nvvm_reflect);
  }

  TLI.addVectorizableFunctionsFromVecLib(ClVectorLibrary);
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl() {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));

  initialize(*this, Triple(), StandardNames);
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl(const Triple &T) {
  // Default to everything being available.
  memset(AvailableArray, -1, sizeof(AvailableArray));

  initialize(*this, T, StandardNames);
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl(const TargetLibraryInfoImpl &TLI)
    : CustomNames(TLI.CustomNames) {
  memcpy(AvailableArray, TLI.AvailableArray, sizeof(AvailableArray));
  VectorDescs = TLI.VectorDescs;
  ScalarDescs = TLI.ScalarDescs;
}

TargetLibraryInfoImpl::TargetLibraryInfoImpl(TargetLibraryInfoImpl &&TLI)
    : CustomNames(std::move(TLI.CustomNames)) {
  std::move(std::begin(TLI.AvailableArray), std::end(TLI.AvailableArray),
            AvailableArray);
  VectorDescs = TLI.VectorDescs;
  ScalarDescs = TLI.ScalarDescs;
}

TargetLibraryInfoImpl &TargetLibraryInfoImpl::operator=(const TargetLibraryInfoImpl &TLI) {
  CustomNames = TLI.CustomNames;
  memcpy(AvailableArray, TLI.AvailableArray, sizeof(AvailableArray));
  return *this;
}

TargetLibraryInfoImpl &TargetLibraryInfoImpl::operator=(TargetLibraryInfoImpl &&TLI) {
  CustomNames = std::move(TLI.CustomNames);
  std::move(std::begin(TLI.AvailableArray), std::end(TLI.AvailableArray),
            AvailableArray);
  return *this;
}

static StringRef sanitizeFunctionName(StringRef funcName) {
  // Filter out empty names and names containing null bytes, those can't be in
  // our table.
  if (funcName.empty() || funcName.find('\0') != StringRef::npos)
    return StringRef();

  // Check for \01 prefix that is used to mangle __asm declarations and
  // strip it if present.
  return GlobalValue::getRealLinkageName(funcName);
}

bool TargetLibraryInfoImpl::getLibFunc(StringRef funcName,
                                       LibFunc::Func &F) const {
  const char *const *Start = &StandardNames[0];
  const char *const *End = &StandardNames[LibFunc::NumLibFuncs];

  funcName = sanitizeFunctionName(funcName);
  if (funcName.empty())
    return false;

  const char *const *I = std::lower_bound(
      Start, End, funcName, [](const char *LHS, StringRef RHS) {
        return std::strncmp(LHS, RHS.data(), RHS.size()) < 0;
      });
  if (I != End && *I == funcName) {
    F = (LibFunc::Func)(I - Start);
    return true;
  }
  return false;
}

bool TargetLibraryInfoImpl::isValidProtoForLibFunc(const FunctionType &FTy,
                                                   LibFunc::Func F,
                                                   const DataLayout *DL) const {
  LLVMContext &Ctx = FTy.getContext();
  Type *PCharTy = Type::getInt8PtrTy(Ctx);
  Type *SizeTTy = DL ? DL->getIntPtrType(Ctx, /*AS=*/0) : nullptr;
  auto IsSizeTTy = [SizeTTy](Type *Ty) {
    return SizeTTy ? Ty == SizeTTy : Ty->isIntegerTy();
  };
  unsigned NumParams = FTy.getNumParams();

  switch (F) {
  case LibFunc::strlen:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getReturnType()->isIntegerTy());

  case LibFunc::strchr:
  case LibFunc::strrchr:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getReturnType() &&
            FTy.getParamType(1)->isIntegerTy());

  case LibFunc::strtol:
  case LibFunc::strtod:
  case LibFunc::strtof:
  case LibFunc::strtoul:
  case LibFunc::strtoll:
  case LibFunc::strtold:
  case LibFunc::strtoull:
    return ((NumParams == 2 || NumParams == 3) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::strcat:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getReturnType() &&
            FTy.getParamType(1) == FTy.getReturnType());

  case LibFunc::strncat:
    return (NumParams == 3 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0) == FTy.getReturnType() &&
            FTy.getParamType(1) == FTy.getReturnType() &&
            FTy.getParamType(2)->isIntegerTy());

  case LibFunc::strcpy_chk:
  case LibFunc::stpcpy_chk:
    --NumParams;
    if (!IsSizeTTy(FTy.getParamType(NumParams)))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc::strcpy:
  case LibFunc::stpcpy:
    return (NumParams == 2 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getParamType(0) == PCharTy);

  case LibFunc::strncpy_chk:
  case LibFunc::stpncpy_chk:
    --NumParams;
    if (!IsSizeTTy(FTy.getParamType(NumParams)))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc::strncpy:
  case LibFunc::stpncpy:
    return (NumParams == 3 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getParamType(0) == PCharTy &&
            FTy.getParamType(2)->isIntegerTy());

  case LibFunc::strxfrm:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  case LibFunc::strcmp:
    return (NumParams == 2 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(0) == FTy.getParamType(1));

  case LibFunc::strncmp:
    return (NumParams == 3 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getParamType(2)->isIntegerTy());

  case LibFunc::strspn:
  case LibFunc::strcspn:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(0) == FTy.getParamType(1) &&
            FTy.getReturnType()->isIntegerTy());

  case LibFunc::strcoll:
  case LibFunc::strcasecmp:
  case LibFunc::strncasecmp:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  case LibFunc::strstr:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  case LibFunc::strpbrk:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0) == FTy.getParamType(1));

  case LibFunc::strtok:
  case LibFunc::strtok_r:
    return (NumParams >= 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::scanf:
  case LibFunc::setbuf:
  case LibFunc::setvbuf:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::strdup:
  case LibFunc::strndup:
    return (NumParams >= 1 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy());
  case LibFunc::sscanf:
  case LibFunc::stat:
  case LibFunc::statvfs:
  case LibFunc::sprintf:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::snprintf:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc::setitimer:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc::system:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::malloc:
    return (NumParams == 1 && FTy.getReturnType()->isPointerTy());
  case LibFunc::memcmp:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getReturnType()->isIntegerTy(32));

  case LibFunc::memchr:
  case LibFunc::memrchr:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy(32) &&
            FTy.getParamType(2)->isIntegerTy() &&
            FTy.getReturnType()->isPointerTy());
  case LibFunc::modf:
  case LibFunc::modff:
  case LibFunc::modfl:
    return (NumParams >= 2 && FTy.getParamType(1)->isPointerTy());

  case LibFunc::memcpy_chk:
  case LibFunc::memmove_chk:
    --NumParams;
    if (!IsSizeTTy(FTy.getParamType(NumParams)))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc::memcpy:
  case LibFunc::mempcpy:
  case LibFunc::memmove:
    return (NumParams == 3 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            IsSizeTTy(FTy.getParamType(2)));

  case LibFunc::memset_chk:
    --NumParams;
    if (!IsSizeTTy(FTy.getParamType(NumParams)))
      return false;
    LLVM_FALLTHROUGH;
  case LibFunc::memset:
    return (NumParams == 3 && FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy() &&
            IsSizeTTy(FTy.getParamType(2)));

  case LibFunc::memccpy:
    return (NumParams >= 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::memalign:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc::realloc:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getReturnType()->isPointerTy());
  case LibFunc::read:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::rewind:
  case LibFunc::rmdir:
  case LibFunc::remove:
  case LibFunc::realpath:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::rename:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::readlink:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::write:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::bcopy:
  case LibFunc::bcmp:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::bzero:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::calloc:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy());

  case LibFunc::atof:
  case LibFunc::atoi:
  case LibFunc::atol:
  case LibFunc::atoll:
  case LibFunc::ferror:
  case LibFunc::getenv:
  case LibFunc::getpwnam:
  case LibFunc::pclose:
  case LibFunc::perror:
  case LibFunc::printf:
  case LibFunc::puts:
  case LibFunc::uname:
  case LibFunc::under_IO_getc:
  case LibFunc::unlink:
  case LibFunc::unsetenv:
    return (NumParams == 1 && FTy.getParamType(0)->isPointerTy());

  case LibFunc::chmod:
  case LibFunc::chown:
  case LibFunc::clearerr:
  case LibFunc::closedir:
  case LibFunc::ctermid:
  case LibFunc::fclose:
  case LibFunc::feof:
  case LibFunc::fflush:
  case LibFunc::fgetc:
  case LibFunc::fileno:
  case LibFunc::flockfile:
  case LibFunc::free:
  case LibFunc::fseek:
  case LibFunc::fseeko64:
  case LibFunc::fseeko:
  case LibFunc::fsetpos:
  case LibFunc::ftell:
  case LibFunc::ftello64:
  case LibFunc::ftello:
  case LibFunc::ftrylockfile:
  case LibFunc::funlockfile:
  case LibFunc::getc:
  case LibFunc::getc_unlocked:
  case LibFunc::getlogin_r:
  case LibFunc::mkdir:
  case LibFunc::mktime:
  case LibFunc::times:
    return (NumParams != 0 && FTy.getParamType(0)->isPointerTy());

  case LibFunc::access:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::fopen:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::fdopen:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::fputc:
  case LibFunc::fstat:
  case LibFunc::frexp:
  case LibFunc::frexpf:
  case LibFunc::frexpl:
  case LibFunc::fstatvfs:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::fgets:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc::fread:
    return (NumParams == 4 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(3)->isPointerTy());
  case LibFunc::fwrite:
    return (NumParams == 4 && FTy.getReturnType()->isIntegerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isIntegerTy() &&
            FTy.getParamType(2)->isIntegerTy() &&
            FTy.getParamType(3)->isPointerTy());
  case LibFunc::fputs:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::fscanf:
  case LibFunc::fprintf:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::fgetpos:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::gets:
  case LibFunc::getchar:
  case LibFunc::getitimer:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::ungetc:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::utime:
  case LibFunc::utimes:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::putc:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::pread:
  case LibFunc::pwrite:
    return (NumParams == 4 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::popen:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::vscanf:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::vsscanf:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc::vfscanf:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc::valloc:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc::vprintf:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::vfprintf:
  case LibFunc::vsprintf:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::vsnprintf:
    return (NumParams == 4 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());
  case LibFunc::open:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::opendir:
    return (NumParams == 1 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy());
  case LibFunc::tmpfile:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc::htonl:
  case LibFunc::htons:
  case LibFunc::ntohl:
  case LibFunc::ntohs:
  case LibFunc::lstat:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::lchown:
    return (NumParams == 3 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::qsort:
    return (NumParams == 4 && FTy.getParamType(3)->isPointerTy());
  case LibFunc::dunder_strdup:
  case LibFunc::dunder_strndup:
    return (NumParams >= 1 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy());
  case LibFunc::dunder_strtok_r:
    return (NumParams == 3 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::under_IO_putc:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::dunder_isoc99_scanf:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::stat64:
  case LibFunc::lstat64:
  case LibFunc::statvfs64:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::dunder_isoc99_sscanf:
    return (NumParams >= 1 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::fopen64:
    return (NumParams == 2 && FTy.getReturnType()->isPointerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());
  case LibFunc::tmpfile64:
    return (FTy.getReturnType()->isPointerTy());
  case LibFunc::fstat64:
  case LibFunc::fstatvfs64:
    return (NumParams == 2 && FTy.getParamType(1)->isPointerTy());
  case LibFunc::open64:
    return (NumParams >= 2 && FTy.getParamType(0)->isPointerTy());
  case LibFunc::gettimeofday:
    return (NumParams == 2 && FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy());

  case LibFunc::Znwj:                    // new(unsigned int);
  case LibFunc::Znwm:                    // new(unsigned long);
  case LibFunc::Znaj:                    // new[](unsigned int);
  case LibFunc::Znam:                    // new[](unsigned long);
  case LibFunc::msvc_new_int:            // new(unsigned int);
  case LibFunc::msvc_new_longlong:       // new(unsigned long long);
  case LibFunc::msvc_new_array_int:      // new[](unsigned int);
  case LibFunc::msvc_new_array_longlong: // new[](unsigned long long);
    return (NumParams == 1);

  case LibFunc::memset_pattern16:
    return (!FTy.isVarArg() && NumParams == 3 &&
            isa<PointerType>(FTy.getParamType(0)) &&
            isa<PointerType>(FTy.getParamType(1)) &&
            isa<IntegerType>(FTy.getParamType(2)));

  // int __nvvm_reflect(const char *);
  case LibFunc::nvvm_reflect:
    return (NumParams == 1 && isa<PointerType>(FTy.getParamType(0)));

  case LibFunc::sin:
  case LibFunc::sinf:
  case LibFunc::sinl:
  case LibFunc::cos:
  case LibFunc::cosf:
  case LibFunc::cosl:
  case LibFunc::tan:
  case LibFunc::tanf:
  case LibFunc::tanl:
  case LibFunc::exp:
  case LibFunc::expf:
  case LibFunc::expl:
  case LibFunc::exp2:
  case LibFunc::exp2f:
  case LibFunc::exp2l:
  case LibFunc::log:
  case LibFunc::logf:
  case LibFunc::logl:
  case LibFunc::log10:
  case LibFunc::log10f:
  case LibFunc::log10l:
  case LibFunc::log2:
  case LibFunc::log2f:
  case LibFunc::log2l:
  case LibFunc::fabs:
  case LibFunc::fabsf:
  case LibFunc::fabsl:
  case LibFunc::floor:
  case LibFunc::floorf:
  case LibFunc::floorl:
  case LibFunc::ceil:
  case LibFunc::ceilf:
  case LibFunc::ceill:
  case LibFunc::trunc:
  case LibFunc::truncf:
  case LibFunc::truncl:
  case LibFunc::rint:
  case LibFunc::rintf:
  case LibFunc::rintl:
  case LibFunc::nearbyint:
  case LibFunc::nearbyintf:
  case LibFunc::nearbyintl:
  case LibFunc::round:
  case LibFunc::roundf:
  case LibFunc::roundl:
  case LibFunc::sqrt:
  case LibFunc::sqrtf:
  case LibFunc::sqrtl:
    return (NumParams == 1 && FTy.getReturnType()->isFloatingPointTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc::fmin:
  case LibFunc::fminf:
  case LibFunc::fminl:
  case LibFunc::fmax:
  case LibFunc::fmaxf:
  case LibFunc::fmaxl:
  case LibFunc::copysign:
  case LibFunc::copysignf:
  case LibFunc::copysignl:
  case LibFunc::pow:
  case LibFunc::powf:
  case LibFunc::powl:
    return (NumParams == 2 && FTy.getReturnType()->isFloatingPointTy() &&
            FTy.getReturnType() == FTy.getParamType(0) &&
            FTy.getReturnType() == FTy.getParamType(1));

  case LibFunc::ffs:
  case LibFunc::ffsl:
  case LibFunc::ffsll:
  case LibFunc::isdigit:
  case LibFunc::isascii:
  case LibFunc::toascii:
    return (NumParams == 1 && FTy.getReturnType()->isIntegerTy(32) &&
            FTy.getParamType(0)->isIntegerTy());

  case LibFunc::fls:
  case LibFunc::flsl:
  case LibFunc::flsll:
  case LibFunc::abs:
  case LibFunc::labs:
  case LibFunc::llabs:
    return (NumParams == 1 && FTy.getReturnType()->isIntegerTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc::cxa_atexit:
    return (NumParams == 3 && FTy.getReturnType()->isIntegerTy() &&
            FTy.getParamType(0)->isPointerTy() &&
            FTy.getParamType(1)->isPointerTy() &&
            FTy.getParamType(2)->isPointerTy());

  case LibFunc::sinpi:
  case LibFunc::cospi:
    return (NumParams == 1 && FTy.getReturnType()->isDoubleTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  case LibFunc::sinpif:
  case LibFunc::cospif:
    return (NumParams == 1 && FTy.getReturnType()->isFloatTy() &&
            FTy.getReturnType() == FTy.getParamType(0));

  default:
    // Assume the other functions are correct.
    // FIXME: It'd be really nice to cover them all.
    return true;
  }
}

bool TargetLibraryInfoImpl::getLibFunc(const Function &FDecl,
                                       LibFunc::Func &F) const {
  const DataLayout *DL =
      FDecl.getParent() ? &FDecl.getParent()->getDataLayout() : nullptr;
  return getLibFunc(FDecl.getName(), F) &&
         isValidProtoForLibFunc(*FDecl.getFunctionType(), F, DL);
}

void TargetLibraryInfoImpl::disableAllFunctions() {
  memset(AvailableArray, 0, sizeof(AvailableArray));
}

static bool compareByScalarFnName(const VecDesc &LHS, const VecDesc &RHS) {
  return std::strncmp(LHS.ScalarFnName, RHS.ScalarFnName,
                      std::strlen(RHS.ScalarFnName)) < 0;
}

static bool compareByVectorFnName(const VecDesc &LHS, const VecDesc &RHS) {
  return std::strncmp(LHS.VectorFnName, RHS.VectorFnName,
                      std::strlen(RHS.VectorFnName)) < 0;
}

static bool compareWithScalarFnName(const VecDesc &LHS, StringRef S) {
  return std::strncmp(LHS.ScalarFnName, S.data(), S.size()) < 0;
}

static bool compareWithVectorFnName(const VecDesc &LHS, StringRef S) {
  return std::strncmp(LHS.VectorFnName, S.data(), S.size()) < 0;
}

void TargetLibraryInfoImpl::addVectorizableFunctions(ArrayRef<VecDesc> Fns) {
  VectorDescs.insert(VectorDescs.end(), Fns.begin(), Fns.end());
  std::sort(VectorDescs.begin(), VectorDescs.end(), compareByScalarFnName);

  ScalarDescs.insert(ScalarDescs.end(), Fns.begin(), Fns.end());
  std::sort(ScalarDescs.begin(), ScalarDescs.end(), compareByVectorFnName);
}

void TargetLibraryInfoImpl::addVectorizableFunctionsFromVecLib(
    enum VectorLibrary VecLib) {
  switch (VecLib) {
  case Accelerate: {
    const VecDesc VecFuncs[] = {
        // Floating-Point Arithmetic and Auxiliary Functions
        {"ceilf", "vceilf", 4},
        {"fabsf", "vfabsf", 4},
        {"llvm.fabs.f32", "vfabsf", 4},
        {"floorf", "vfloorf", 4},
        {"sqrtf", "vsqrtf", 4},
        {"llvm.sqrt.f32", "vsqrtf", 4},

        // Exponential and Logarithmic Functions
        {"expf", "vexpf", 4},
        {"llvm.exp.f32", "vexpf", 4},
        {"expm1f", "vexpm1f", 4},
        {"logf", "vlogf", 4},
        {"llvm.log.f32", "vlogf", 4},
        {"log1pf", "vlog1pf", 4},
        {"log10f", "vlog10f", 4},
        {"llvm.log10.f32", "vlog10f", 4},
        {"logbf", "vlogbf", 4},

        // Trigonometric Functions
        {"sinf", "vsinf", 4},
        {"llvm.sin.f32", "vsinf", 4},
        {"cosf", "vcosf", 4},
        {"llvm.cos.f32", "vcosf", 4},
        {"tanf", "vtanf", 4},
        {"asinf", "vasinf", 4},
        {"acosf", "vacosf", 4},
        {"atanf", "vatanf", 4},

        // Hyperbolic Functions
        {"sinhf", "vsinhf", 4},
        {"coshf", "vcoshf", 4},
        {"tanhf", "vtanhf", 4},
        {"asinhf", "vasinhf", 4},
        {"acoshf", "vacoshf", 4},
        {"atanhf", "vatanhf", 4},
    };
    addVectorizableFunctions(VecFuncs);
    break;
  }
  case SVML: {
    const VecDesc VecFuncs[] = {
        {"sin", "__svml_sin2", 2},
        {"sin", "__svml_sin4", 4},
        {"sin", "__svml_sin8", 8},

        {"sinf", "__svml_sinf4", 4},
        {"sinf", "__svml_sinf8", 8},
        {"sinf", "__svml_sinf16", 16},

        {"cos", "__svml_cos2", 2},
        {"cos", "__svml_cos4", 4},
        {"cos", "__svml_cos8", 8},

        {"cosf", "__svml_cosf4", 4},
        {"cosf", "__svml_cosf8", 8},
        {"cosf", "__svml_cosf16", 16},

        {"pow", "__svml_pow2", 2},
        {"pow", "__svml_pow4", 4},
        {"pow", "__svml_pow8", 8},

        {"powf", "__svml_powf4", 4},
        {"powf", "__svml_powf8", 8},
        {"powf", "__svml_powf16", 16},

        {"llvm.pow.f64", "__svml_pow2", 2},
        {"llvm.pow.f64", "__svml_pow4", 4},
        {"llvm.pow.f64", "__svml_pow8", 8},

        {"llvm.pow.f32", "__svml_powf4", 4},
        {"llvm.pow.f32", "__svml_powf8", 8},
        {"llvm.pow.f32", "__svml_powf16", 16},

        {"exp", "__svml_exp2", 2},
        {"exp", "__svml_exp4", 4},
        {"exp", "__svml_exp8", 8},

        {"expf", "__svml_expf4", 4},
        {"expf", "__svml_expf8", 8},
        {"expf", "__svml_expf16", 16},

        {"llvm.exp.f64", "__svml_exp2", 2},
        {"llvm.exp.f64", "__svml_exp4", 4},
        {"llvm.exp.f64", "__svml_exp8", 8},

        {"llvm.exp.f32", "__svml_expf4", 4},
        {"llvm.exp.f32", "__svml_expf8", 8},
        {"llvm.exp.f32", "__svml_expf16", 16},

        {"log", "__svml_log2", 2},
        {"log", "__svml_log4", 4},
        {"log", "__svml_log8", 8},

        {"logf", "__svml_logf4", 4},
        {"logf", "__svml_logf8", 8},
        {"logf", "__svml_logf16", 16},

        {"llvm.log.f64", "__svml_log2", 2},
        {"llvm.log.f64", "__svml_log4", 4},
        {"llvm.log.f64", "__svml_log8", 8},

        {"llvm.log.f32", "__svml_logf4", 4},
        {"llvm.log.f32", "__svml_logf8", 8},
        {"llvm.log.f32", "__svml_logf16", 16},
    };
    addVectorizableFunctions(VecFuncs);
    break;
  }
  case NoLibrary:
    break;
  }
}

bool TargetLibraryInfoImpl::isFunctionVectorizable(StringRef funcName) const {
  funcName = sanitizeFunctionName(funcName);
  if (funcName.empty())
    return false;

  std::vector<VecDesc>::const_iterator I = std::lower_bound(
      VectorDescs.begin(), VectorDescs.end(), funcName,
      compareWithScalarFnName);
  return I != VectorDescs.end() && StringRef(I->ScalarFnName) == funcName;
}

StringRef TargetLibraryInfoImpl::getVectorizedFunction(StringRef F,
                                                       unsigned VF) const {
  F = sanitizeFunctionName(F);
  if (F.empty())
    return F;
  std::vector<VecDesc>::const_iterator I = std::lower_bound(
      VectorDescs.begin(), VectorDescs.end(), F, compareWithScalarFnName);
  while (I != VectorDescs.end() && StringRef(I->ScalarFnName) == F) {
    if (I->VectorizationFactor == VF)
      return I->VectorFnName;
    ++I;
  }
  return StringRef();
}

StringRef TargetLibraryInfoImpl::getScalarizedFunction(StringRef F,
                                                       unsigned &VF) const {
  F = sanitizeFunctionName(F);
  if (F.empty())
    return F;

  std::vector<VecDesc>::const_iterator I = std::lower_bound(
      ScalarDescs.begin(), ScalarDescs.end(), F, compareWithVectorFnName);
  if (I == VectorDescs.end() || StringRef(I->VectorFnName) != F)
    return StringRef();
  VF = I->VectorizationFactor;
  return I->ScalarFnName;
}

TargetLibraryInfo TargetLibraryAnalysis::run(Module &M,
                                             ModuleAnalysisManager &) {
  if (PresetInfoImpl)
    return TargetLibraryInfo(*PresetInfoImpl);

  return TargetLibraryInfo(lookupInfoImpl(Triple(M.getTargetTriple())));
}

TargetLibraryInfo TargetLibraryAnalysis::run(Function &F,
                                             FunctionAnalysisManager &) {
  if (PresetInfoImpl)
    return TargetLibraryInfo(*PresetInfoImpl);

  return TargetLibraryInfo(
      lookupInfoImpl(Triple(F.getParent()->getTargetTriple())));
}

TargetLibraryInfoImpl &TargetLibraryAnalysis::lookupInfoImpl(const Triple &T) {
  std::unique_ptr<TargetLibraryInfoImpl> &Impl =
      Impls[T.normalize()];
  if (!Impl)
    Impl.reset(new TargetLibraryInfoImpl(T));

  return *Impl;
}


TargetLibraryInfoWrapperPass::TargetLibraryInfoWrapperPass()
    : ImmutablePass(ID), TLIImpl(), TLI(TLIImpl) {
  initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

TargetLibraryInfoWrapperPass::TargetLibraryInfoWrapperPass(const Triple &T)
    : ImmutablePass(ID), TLIImpl(T), TLI(TLIImpl) {
  initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

TargetLibraryInfoWrapperPass::TargetLibraryInfoWrapperPass(
    const TargetLibraryInfoImpl &TLIImpl)
    : ImmutablePass(ID), TLIImpl(TLIImpl), TLI(this->TLIImpl) {
  initializeTargetLibraryInfoWrapperPassPass(*PassRegistry::getPassRegistry());
}

char TargetLibraryAnalysis::PassID;

// Register the basic pass.
INITIALIZE_PASS(TargetLibraryInfoWrapperPass, "targetlibinfo",
                "Target Library Information", false, true)
char TargetLibraryInfoWrapperPass::ID = 0;

void TargetLibraryInfoWrapperPass::anchor() {}
