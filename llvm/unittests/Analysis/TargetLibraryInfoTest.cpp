//===--- TargetLibraryInfoTest.cpp - TLI/LibFunc unit tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TargetLibraryInfoTest : public testing::Test {
protected:
  LLVMContext Context;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;

  std::unique_ptr<Module> M;

  TargetLibraryInfoTest() : TLI(TLII) {}

  void parseAssembly(const char *Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    if (!M)
      report_fatal_error(os.str());
  }

  ::testing::AssertionResult isLibFunc(const Function *FDecl,
                                       LibFunc ExpectedLF) {
    StringRef ExpectedLFName = TLI.getName(ExpectedLF);

    if (!FDecl)
      return ::testing::AssertionFailure() << ExpectedLFName << " not found";

    LibFunc F;
    if (!TLI.getLibFunc(*FDecl, F))
      return ::testing::AssertionFailure() << ExpectedLFName << " invalid";

    return ::testing::AssertionSuccess() << ExpectedLFName << " is LibFunc";
  }
};

} // end anonymous namespace

// Check that we don't accept egregiously incorrect prototypes.
TEST_F(TargetLibraryInfoTest, InvalidProto) {
  parseAssembly("%foo = type { %foo }\n");

  auto *StructTy = M->getTypeByName("foo");
  auto *InvalidFTy = FunctionType::get(StructTy, /*isVarArg=*/false);

  for (unsigned FI = 0; FI != LibFunc::NumLibFuncs; ++FI) {
    LibFunc LF = (LibFunc)FI;
    auto *F = cast<Function>(
        M->getOrInsertFunction(TLI.getName(LF), InvalidFTy));
    EXPECT_FALSE(isLibFunc(F, LF));
  }
}

// Check that we do accept know-correct prototypes.
TEST_F(TargetLibraryInfoTest, ValidProto) {
  parseAssembly(
    // These functions use a 64-bit size_t; use the appropriate datalayout.
    "target datalayout = \"p:64:64:64\"\n"

    // Struct pointers are replaced with an opaque pointer.
    "%struct = type opaque\n"

    // These functions were extracted as-is from the OS X headers.
    "declare double @__cospi(double)\n"
    "declare float @__cospif(float)\n"
    "declare { double, double } @__sincospi_stret(double)\n"
    "declare <2 x float> @__sincospif_stret(float)\n"
    "declare double @__sinpi(double)\n"
    "declare float @__sinpif(float)\n"
    "declare i32 @abs(i32)\n"
    "declare i32 @access(i8*, i32)\n"
    "declare double @acos(double)\n"
    "declare float @acosf(float)\n"
    "declare double @acosh(double)\n"
    "declare float @acoshf(float)\n"
    "declare x86_fp80 @acoshl(x86_fp80)\n"
    "declare x86_fp80 @acosl(x86_fp80)\n"
    "declare double @asin(double)\n"
    "declare float @asinf(float)\n"
    "declare double @asinh(double)\n"
    "declare float @asinhf(float)\n"
    "declare x86_fp80 @asinhl(x86_fp80)\n"
    "declare x86_fp80 @asinl(x86_fp80)\n"
    "declare double @atan(double)\n"
    "declare double @atan2(double, double)\n"
    "declare float @atan2f(float, float)\n"
    "declare x86_fp80 @atan2l(x86_fp80, x86_fp80)\n"
    "declare float @atanf(float)\n"
    "declare double @atanh(double)\n"
    "declare float @atanhf(float)\n"
    "declare x86_fp80 @atanhl(x86_fp80)\n"
    "declare x86_fp80 @atanl(x86_fp80)\n"
    "declare double @atof(i8*)\n"
    "declare i32 @atoi(i8*)\n"
    "declare i64 @atol(i8*)\n"
    "declare i64 @atoll(i8*)\n"
    "declare i32 @bcmp(i8*, i8*, i64)\n"
    "declare void @bcopy(i8*, i8*, i64)\n"
    "declare void @bzero(i8*, i64)\n"
    "declare i8* @calloc(i64, i64)\n"
    "declare double @cbrt(double)\n"
    "declare float @cbrtf(float)\n"
    "declare x86_fp80 @cbrtl(x86_fp80)\n"
    "declare double @ceil(double)\n"
    "declare float @ceilf(float)\n"
    "declare x86_fp80 @ceill(x86_fp80)\n"
    "declare i32 @chown(i8*, i32, i32)\n"
    "declare void @clearerr(%struct*)\n"
    "declare double @copysign(double, double)\n"
    "declare float @copysignf(float, float)\n"
    "declare x86_fp80 @copysignl(x86_fp80, x86_fp80)\n"
    "declare double @cos(double)\n"
    "declare float @cosf(float)\n"
    "declare double @cosh(double)\n"
    "declare float @coshf(float)\n"
    "declare x86_fp80 @coshl(x86_fp80)\n"
    "declare x86_fp80 @cosl(x86_fp80)\n"
    "declare i8* @ctermid(i8*)\n"
    "declare double @exp(double)\n"
    "declare double @exp2(double)\n"
    "declare float @exp2f(float)\n"
    "declare x86_fp80 @exp2l(x86_fp80)\n"
    "declare float @expf(float)\n"
    "declare x86_fp80 @expl(x86_fp80)\n"
    "declare double @expm1(double)\n"
    "declare float @expm1f(float)\n"
    "declare x86_fp80 @expm1l(x86_fp80)\n"
    "declare double @fabs(double)\n"
    "declare float @fabsf(float)\n"
    "declare x86_fp80 @fabsl(x86_fp80)\n"
    "declare i32 @fclose(%struct*)\n"
    "declare i32 @feof(%struct*)\n"
    "declare i32 @ferror(%struct*)\n"
    "declare i32 @fflush(%struct*)\n"
    "declare i32 @ffs(i32)\n"
    "declare i32 @ffsl(i64)\n"
    "declare i32 @ffsll(i64)\n"
    "declare i32 @fgetc(%struct*)\n"
    "declare i32 @fgetpos(%struct*, i64*)\n"
    "declare i8* @fgets(i8*, i32, %struct*)\n"
    "declare i32 @fileno(%struct*)\n"
    "declare void @flockfile(%struct*)\n"
    "declare double @floor(double)\n"
    "declare float @floorf(float)\n"
    "declare x86_fp80 @floorl(x86_fp80)\n"
    "declare i32 @fls(i32)\n"
    "declare i32 @flsl(i64)\n"
    "declare i32 @flsll(i64)\n"
    "declare double @fmax(double, double)\n"
    "declare float @fmaxf(float, float)\n"
    "declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)\n"
    "declare double @fmin(double, double)\n"
    "declare float @fminf(float, float)\n"
    "declare x86_fp80 @fminl(x86_fp80, x86_fp80)\n"
    "declare double @fmod(double, double)\n"
    "declare float @fmodf(float, float)\n"
    "declare x86_fp80 @fmodl(x86_fp80, x86_fp80)\n"
    "declare i32 @fprintf(%struct*, i8*, ...)\n"
    "declare i32 @fputc(i32, %struct*)\n"
    "declare i64 @fread(i8*, i64, i64, %struct*)\n"
    "declare void @free(i8*)\n"
    "declare double @frexp(double, i32*)\n"
    "declare float @frexpf(float, i32*)\n"
    "declare x86_fp80 @frexpl(x86_fp80, i32*)\n"
    "declare i32 @fscanf(%struct*, i8*, ...)\n"
    "declare i32 @fseek(%struct*, i64, i32)\n"
    "declare i32 @fseeko(%struct*, i64, i32)\n"
    "declare i32 @fsetpos(%struct*, i64*)\n"
    "declare i32 @fstatvfs(i32, %struct*)\n"
    "declare i64 @ftell(%struct*)\n"
    "declare i64 @ftello(%struct*)\n"
    "declare i32 @ftrylockfile(%struct*)\n"
    "declare void @funlockfile(%struct*)\n"
    "declare i32 @getc(%struct*)\n"
    "declare i32 @getc_unlocked(%struct*)\n"
    "declare i32 @getchar()\n"
    "declare i8* @getenv(i8*)\n"
    "declare i32 @getitimer(i32, %struct*)\n"
    "declare i32 @getlogin_r(i8*, i64)\n"
    "declare %struct* @getpwnam(i8*)\n"
    "declare i8* @gets(i8*)\n"
    "declare i32 @gettimeofday(%struct*, i8*)\n"
    "declare i32 @_Z7isasciii(i32)\n"
    "declare i32 @_Z7isdigiti(i32)\n"
    "declare i64 @labs(i64)\n"
    "declare double @ldexp(double, i32)\n"
    "declare float @ldexpf(float, i32)\n"
    "declare x86_fp80 @ldexpl(x86_fp80, i32)\n"
    "declare i64 @llabs(i64)\n"
    "declare double @log(double)\n"
    "declare double @log10(double)\n"
    "declare float @log10f(float)\n"
    "declare x86_fp80 @log10l(x86_fp80)\n"
    "declare double @log1p(double)\n"
    "declare float @log1pf(float)\n"
    "declare x86_fp80 @log1pl(x86_fp80)\n"
    "declare double @log2(double)\n"
    "declare float @log2f(float)\n"
    "declare x86_fp80 @log2l(x86_fp80)\n"
    "declare double @logb(double)\n"
    "declare float @logbf(float)\n"
    "declare x86_fp80 @logbl(x86_fp80)\n"
    "declare float @logf(float)\n"
    "declare x86_fp80 @logl(x86_fp80)\n"
    "declare i8* @malloc(i64)\n"
    "declare i8* @memccpy(i8*, i8*, i32, i64)\n"
    "declare i8* @memchr(i8*, i32, i64)\n"
    "declare i32 @memcmp(i8*, i8*, i64)\n"
    "declare i8* @memcpy(i8*, i8*, i64)\n"
    "declare i8* @memmove(i8*, i8*, i64)\n"
    "declare i8* @memset(i8*, i32, i64)\n"
    "declare void @memset_pattern16(i8*, i8*, i64)\n"
    "declare i32 @mkdir(i8*, i16)\n"
    "declare double @modf(double, double*)\n"
    "declare float @modff(float, float*)\n"
    "declare x86_fp80 @modfl(x86_fp80, x86_fp80*)\n"
    "declare double @nearbyint(double)\n"
    "declare float @nearbyintf(float)\n"
    "declare x86_fp80 @nearbyintl(x86_fp80)\n"
    "declare i32 @pclose(%struct*)\n"
    "declare void @perror(i8*)\n"
    "declare i32 @posix_memalign(i8**, i64, i64)\n"
    "declare double @pow(double, double)\n"
    "declare float @powf(float, float)\n"
    "declare x86_fp80 @powl(x86_fp80, x86_fp80)\n"
    "declare i32 @printf(i8*, ...)\n"
    "declare i32 @putc(i32, %struct*)\n"
    "declare i32 @putchar(i32)\n"
    "declare i32 @puts(i8*)\n"
    "declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)*)\n"
    "declare i64 @readlink(i8*, i8*, i64)\n"
    "declare i8* @realloc(i8*, i64)\n"
    "declare i8* @reallocf(i8*, i64)\n"
    "declare i32 @remove(i8*)\n"
    "declare i32 @rename(i8*, i8*)\n"
    "declare void @rewind(%struct*)\n"
    "declare double @rint(double)\n"
    "declare float @rintf(float)\n"
    "declare x86_fp80 @rintl(x86_fp80)\n"
    "declare i32 @rmdir(i8*)\n"
    "declare double @round(double)\n"
    "declare float @roundf(float)\n"
    "declare x86_fp80 @roundl(x86_fp80)\n"
    "declare i32 @scanf(i8*, ...)\n"
    "declare void @setbuf(%struct*, i8*)\n"
    "declare i32 @setitimer(i32, %struct*, %struct*)\n"
    "declare i32 @setvbuf(%struct*, i8*, i32, i64)\n"
    "declare double @sin(double)\n"
    "declare float @sinf(float)\n"
    "declare double @sinh(double)\n"
    "declare float @sinhf(float)\n"
    "declare x86_fp80 @sinhl(x86_fp80)\n"
    "declare x86_fp80 @sinl(x86_fp80)\n"
    "declare i32 @snprintf(i8*, i64, i8*, ...)\n"
    "declare i32 @sprintf(i8*, i8*, ...)\n"
    "declare double @sqrt(double)\n"
    "declare float @sqrtf(float)\n"
    "declare x86_fp80 @sqrtl(x86_fp80)\n"
    "declare i32 @sscanf(i8*, i8*, ...)\n"
    "declare i32 @statvfs(i8*, %struct*)\n"
    "declare i8* @stpcpy(i8*, i8*)\n"
    "declare i8* @stpncpy(i8*, i8*, i64)\n"
    "declare i32 @strcasecmp(i8*, i8*)\n"
    "declare i8* @strcat(i8*, i8*)\n"
    "declare i8* @strchr(i8*, i32)\n"
    "declare i32 @strcmp(i8*, i8*)\n"
    "declare i32 @strcoll(i8*, i8*)\n"
    "declare i8* @strcpy(i8*, i8*)\n"
    "declare i64 @strcspn(i8*, i8*)\n"
    "declare i8* @strdup(i8*)\n"
    "declare i64 @strlen(i8*)\n"
    "declare i32 @strncasecmp(i8*, i8*, i64)\n"
    "declare i8* @strncat(i8*, i8*, i64)\n"
    "declare i32 @strncmp(i8*, i8*, i64)\n"
    "declare i8* @strncpy(i8*, i8*, i64)\n"
    "declare i8* @strndup(i8*, i64)\n"
    "declare i64 @strnlen(i8*, i64)\n"
    "declare i8* @strpbrk(i8*, i8*)\n"
    "declare i8* @strrchr(i8*, i32)\n"
    "declare i64 @strspn(i8*, i8*)\n"
    "declare i8* @strstr(i8*, i8*)\n"
    "declare i8* @strtok(i8*, i8*)\n"
    "declare i8* @strtok_r(i8*, i8*, i8**)\n"
    "declare i64 @strtol(i8*, i8**, i32)\n"
    "declare x86_fp80 @strtold(i8*, i8**)\n"
    "declare i64 @strtoll(i8*, i8**, i32)\n"
    "declare i64 @strtoul(i8*, i8**, i32)\n"
    "declare i64 @strtoull(i8*, i8**, i32)\n"
    "declare i64 @strxfrm(i8*, i8*, i64)\n"
    "declare double @tan(double)\n"
    "declare float @tanf(float)\n"
    "declare double @tanh(double)\n"
    "declare float @tanhf(float)\n"
    "declare x86_fp80 @tanhl(x86_fp80)\n"
    "declare x86_fp80 @tanl(x86_fp80)\n"
    "declare i64 @times(%struct*)\n"
    "declare %struct* @tmpfile()\n"
    "declare i32 @_Z7toasciii(i32)\n"
    "declare double @trunc(double)\n"
    "declare float @truncf(float)\n"
    "declare x86_fp80 @truncl(x86_fp80)\n"
    "declare i32 @uname(%struct*)\n"
    "declare i32 @ungetc(i32, %struct*)\n"
    "declare i32 @unlink(i8*)\n"
    "declare i32 @utime(i8*, %struct*)\n"
    "declare i32 @utimes(i8*, %struct*)\n"
    "declare i8* @valloc(i64)\n"
    "declare i32 @vfprintf(%struct*, i8*, %struct*)\n"
    "declare i32 @vfscanf(%struct*, i8*, %struct*)\n"
    "declare i32 @vprintf(i8*, %struct*)\n"
    "declare i32 @vscanf(i8*, %struct*)\n"
    "declare i32 @vsnprintf(i8*, i64, i8*, %struct*)\n"
    "declare i32 @vsprintf(i8*, i8*, %struct*)\n"
    "declare i32 @vsscanf(i8*, i8*, %struct*)\n"
    "declare i64 @wcslen(i32*)\n"

    // These functions were also extracted from the OS X headers, but they are
    // available with a special name on darwin.
    // This test uses the default TLI name instead.
    "declare i32 @chmod(i8*, i16)\n"
    "declare i32 @closedir(%struct*)\n"
    "declare %struct* @fdopen(i32, i8*)\n"
    "declare %struct* @fopen(i8*, i8*)\n"
    "declare i32 @fputs(i8*, %struct*)\n"
    "declare i32 @fstat(i32, %struct*)\n"
    "declare i64 @fwrite(i8*, i64, i64, %struct*)\n"
    "declare i32 @lchown(i8*, i32, i32)\n"
    "declare i32 @lstat(i8*, %struct*)\n"
    "declare i64 @mktime(%struct*)\n"
    "declare i32 @open(i8*, i32, ...)\n"
    "declare %struct* @opendir(i8*)\n"
    "declare %struct* @popen(i8*, i8*)\n"
    "declare i64 @pread(i32, i8*, i64, i64)\n"
    "declare i64 @pwrite(i32, i8*, i64, i64)\n"
    "declare i64 @read(i32, i8*, i64)\n"
    "declare i8* @realpath(i8*, i8*)\n"
    "declare i32 @stat(i8*, %struct*)\n"
    "declare double @strtod(i8*, i8**)\n"
    "declare float @strtof(i8*, i8**)\n"
    "declare i32 @system(i8*)\n"
    "declare i32 @unsetenv(i8*)\n"
    "declare i64 @write(i32, i8*, i64)\n"

    // These functions are available on Linux but not Darwin; they only differ
    // from their non-64 counterparts in the struct type.
    // Use the same prototype as the non-64 variant.
    "declare %struct* @fopen64(i8*, i8*)\n"
    "declare i32 @fstat64(i32, %struct*)\n"
    "declare i32 @fstatvfs64(i32, %struct*)\n"
    "declare i32 @lstat64(i8*, %struct*)\n"
    "declare i32 @open64(i8*, i32, ...)\n"
    "declare i32 @stat64(i8*, %struct*)\n"
    "declare i32 @statvfs64(i8*, %struct*)\n"
    "declare %struct* @tmpfile64()\n"

    // These functions are also -64 variants, but do differ in the type of the
    // off_t (vs off64_t) parameter.  The non-64 variants declared above used
    // a 64-bit off_t, so, in practice, they are also equivalent.
    "declare i32 @fseeko64(%struct*, i64, i32)\n"
    "declare i64 @ftello64(%struct*)\n"

    "declare void @_ZdaPv(i8*)\n"
    "declare void @_ZdaPvRKSt9nothrow_t(i8*, %struct*)\n"
    "declare void @_ZdaPvj(i8*, i32)\n"
    "declare void @_ZdaPvm(i8*, i64)\n"
    "declare void @_ZdlPv(i8*)\n"
    "declare void @_ZdlPvRKSt9nothrow_t(i8*, %struct*)\n"
    "declare void @_ZdlPvj(i8*, i32)\n"
    "declare void @_ZdlPvm(i8*, i64)\n"
    "declare i8* @_Znaj(i32)\n"
    "declare i8* @_ZnajRKSt9nothrow_t(i32, %struct*)\n"
    "declare i8* @_Znam(i64)\n"
    "declare i8* @_ZnamRKSt9nothrow_t(i64, %struct*)\n"
    "declare i8* @_Znwj(i32)\n"
    "declare i8* @_ZnwjRKSt9nothrow_t(i32, %struct*)\n"
    "declare i8* @_Znwm(i64)\n"
    "declare i8* @_ZnwmRKSt9nothrow_t(i64, %struct*)\n"

    "declare void @\"??3@YAXPEAX@Z\"(i8*)\n"
    "declare void @\"??3@YAXPEAXAEBUnothrow_t@std@@@Z\"(i8*, %struct*)\n"
    "declare void @\"??3@YAXPEAX_K@Z\"(i8*, i64)\n"
    "declare void @\"??_V@YAXPEAX@Z\"(i8*)\n"
    "declare void @\"??_V@YAXPEAXAEBUnothrow_t@std@@@Z\"(i8*, %struct*)\n"
    "declare void @\"??_V@YAXPEAX_K@Z\"(i8*, i64)\n"
    "declare i8* @\"??2@YAPAXI@Z\"(i32)\n"
    "declare i8* @\"??2@YAPAXIABUnothrow_t@std@@@Z\"(i32, %struct*)\n"
    "declare i8* @\"??2@YAPEAX_K@Z\"(i64)\n"
    "declare i8* @\"??2@YAPEAX_KAEBUnothrow_t@std@@@Z\"(i64, %struct*)\n"
    "declare i8* @\"??_U@YAPAXI@Z\"(i32)\n"
    "declare i8* @\"??_U@YAPAXIABUnothrow_t@std@@@Z\"(i32, %struct*)\n"
    "declare i8* @\"??_U@YAPEAX_K@Z\"(i64)\n"
    "declare i8* @\"??_U@YAPEAX_KAEBUnothrow_t@std@@@Z\"(i64, %struct*)\n"

    "declare void @\"??3@YAXPAX@Z\"(i8*)\n"
    "declare void @\"??3@YAXPAXABUnothrow_t@std@@@Z\"(i8*, %struct*)\n"
    "declare void @\"??3@YAXPAXI@Z\"(i8*, i32)\n"
    "declare void @\"??_V@YAXPAX@Z\"(i8*)\n"
    "declare void @\"??_V@YAXPAXABUnothrow_t@std@@@Z\"(i8*, %struct*)\n"
    "declare void @\"??_V@YAXPAXI@Z\"(i8*, i32)\n"

    // These other functions were derived from the .def C declaration.
    "declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*)\n"
    "declare void @__cxa_guard_abort(%struct*)\n"
    "declare i32 @__cxa_guard_acquire(%struct*)\n"
    "declare void @__cxa_guard_release(%struct*)\n"

    "declare i32 @__nvvm_reflect(i8*)\n"

    "declare i8* @__memcpy_chk(i8*, i8*, i64, i64)\n"
    "declare i8* @__memmove_chk(i8*, i8*, i64, i64)\n"
    "declare i8* @__memset_chk(i8*, i32, i64, i64)\n"
    "declare i8* @__stpcpy_chk(i8*, i8*, i64)\n"
    "declare i8* @__stpncpy_chk(i8*, i8*, i64, i64)\n"
    "declare i8* @__strcpy_chk(i8*, i8*, i64)\n"
    "declare i8* @__strncpy_chk(i8*, i8*, i64, i64)\n"

    "declare i8* @memalign(i64, i64)\n"
    "declare i8* @mempcpy(i8*, i8*, i64)\n"
    "declare i8* @memrchr(i8*, i32, i64)\n"

    // These are similar to the FILE* fgetc/fputc.
    "declare i32 @_IO_getc(%struct*)\n"
    "declare i32 @_IO_putc(i32, %struct*)\n"

    "declare i32 @__isoc99_scanf(i8*, ...)\n"
    "declare i32 @__isoc99_sscanf(i8*, i8*, ...)\n"
    "declare i8* @__strdup(i8*)\n"
    "declare i8* @__strndup(i8*, i64)\n"
    "declare i8* @__strtok_r(i8*, i8*, i8**)\n"

    "declare double @__sqrt_finite(double)\n"
    "declare float @__sqrtf_finite(float)\n"
    "declare x86_fp80 @__sqrtl_finite(x86_fp80)\n"
    "declare double @exp10(double)\n"
    "declare float @exp10f(float)\n"
    "declare x86_fp80 @exp10l(x86_fp80)\n"

    // These printf variants have the same prototype as the non-'i' versions.
    "declare i32 @fiprintf(%struct*, i8*, ...)\n"
    "declare i32 @iprintf(i8*, ...)\n"
    "declare i32 @siprintf(i8*, i8*, ...)\n"

    "declare i32 @htonl(i32)\n"
    "declare i16 @htons(i16)\n"
    "declare i32 @ntohl(i32)\n"
    "declare i16 @ntohs(i16)\n"

    "declare i32 @isascii(i32)\n"
    "declare i32 @isdigit(i32)\n"
    "declare i32 @toascii(i32)\n"

    // These functions were extracted from math-finite.h which provides
    // functions similar to those in math.h, but optimized for handling
    // finite values only.
    "declare double @__acos_finite(double)\n"
    "declare float @__acosf_finite(float)\n"
    "declare x86_fp80 @__acosl_finite(x86_fp80)\n"
    "declare double @__acosh_finite(double)\n"
    "declare float @__acoshf_finite(float)\n"
    "declare x86_fp80 @__acoshl_finite(x86_fp80)\n"
    "declare double @__asin_finite(double)\n"
    "declare float @__asinf_finite(float)\n"
    "declare x86_fp80 @__asinl_finite(x86_fp80)\n"
    "declare double @__atan2_finite(double, double)\n"
    "declare float @__atan2f_finite(float, float)\n"
    "declare x86_fp80 @__atan2l_finite(x86_fp80, x86_fp80)\n"
    "declare double @__atanh_finite(double)\n"
    "declare float @__atanhf_finite(float)\n"
    "declare x86_fp80 @__atanhl_finite(x86_fp80)\n"
    "declare double @__cosh_finite(double)\n"
    "declare float @__coshf_finite(float)\n"
    "declare x86_fp80 @__coshl_finite(x86_fp80)\n"
    "declare double @__exp10_finite(double)\n"
    "declare float @__exp10f_finite(float)\n"
    "declare x86_fp80 @__exp10l_finite(x86_fp80)\n"
    "declare double @__exp2_finite(double)\n"
    "declare float @__exp2f_finite(float)\n"
    "declare x86_fp80 @__exp2l_finite(x86_fp80)\n"
    "declare double @__exp_finite(double)\n"
    "declare float @__expf_finite(float)\n"
    "declare x86_fp80 @__expl_finite(x86_fp80)\n"     
    "declare double @__log10_finite(double)\n"
    "declare float @__log10f_finite(float)\n"
    "declare x86_fp80 @__log10l_finite(x86_fp80)\n"
    "declare double @__log2_finite(double)\n"
    "declare float @__log2f_finite(float)\n"
    "declare x86_fp80 @__log2l_finite(x86_fp80)\n"
    "declare double @__log_finite(double)\n"
    "declare float @__logf_finite(float)\n"
    "declare x86_fp80 @__logl_finite(x86_fp80)\n"
    "declare double @__pow_finite(double, double)\n"
    "declare float @__powf_finite(float, float)\n"
    "declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80)\n"
    "declare double @__sinh_finite(double)\n"
    "declare float @__sinhf_finite(float)\n"
    "declare x86_fp80 @__sinhl_finite(x86_fp80)\n"
    );

  for (unsigned FI = 0; FI != LibFunc::NumLibFuncs; ++FI) {
    LibFunc LF = (LibFunc)FI;
    // Make sure everything is available; we're not testing target defaults.
    TLII.setAvailable(LF);
    Function *F = M->getFunction(TLI.getName(LF));
    EXPECT_TRUE(isLibFunc(F, LF));
  }
}
