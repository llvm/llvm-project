//===--- TargetLibraryInfoTest.cpp - TLI/LibFunc unit tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
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

  TargetLibraryInfoTest() : TLII(Triple()), TLI(TLII) {}

  void parseAssembly(const char *Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    if (!M)
      report_fatal_error(Twine(errMsg));
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
  parseAssembly("%foo = type opaque\n");

  auto *StructTy = StructType::getTypeByName(Context, "foo");
  auto *InvalidFTy = FunctionType::get(StructTy, /*isVarArg=*/false);

  for (unsigned FI = LibFunc::Begin_LibFunc; FI != LibFunc::End_LibFunc; ++FI) {
    LibFunc LF = (LibFunc)FI;
    auto *F = cast<Function>(
        M->getOrInsertFunction(TLI.getName(LF), InvalidFTy).getCallee());
    EXPECT_FALSE(isLibFunc(F, LF));
  }

  // i64 @labs(i32)
  {
    auto *InvalidLabsFTy = FunctionType::get(Type::getInt64Ty(Context),
                                             {Type::getInt32Ty(Context)},
                                             /*isVarArg=*/false);
    auto *F = cast<Function>(
        M->getOrInsertFunction("labs", InvalidLabsFTy).getCallee());
    EXPECT_FALSE(isLibFunc(F, LibFunc_labs));
  }
}

TEST_F(TargetLibraryInfoTest, SizeReturningNewInvalidProto) {
  parseAssembly(
      "target datalayout = \"p:64:64:64\"\n"
      ";; Invalid additional params \n"
      "declare {ptr, i64} @__size_returning_new(i64, i64)\n"
      ";; Invalid params types \n"
      "declare {ptr, i64} @__size_returning_new_hot_cold(i64, i32)\n"
      ";; Invalid return struct types \n"
      "declare {ptr, i8} @__size_returning_new_aligned(i64, i64)\n"
      ";; Invalid return type \n"
      "declare ptr @__size_returning_new_aligned_hot_cold(i64, i64, i8)\n");

  for (const LibFunc LF :
       {LibFunc_size_returning_new, LibFunc_size_returning_new_aligned,
        LibFunc_size_returning_new_hot_cold,
        LibFunc_size_returning_new_aligned_hot_cold}) {
    TLII.setAvailable(LF);
    Function *F = M->getFunction(TLI.getName(LF));
    ASSERT_NE(F, nullptr);
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
      "declare i32 @access(ptr, i32)\n"
      "declare double @acos(double)\n"
      "declare float @acosf(float)\n"
      "declare double @acosh(double)\n"
      "declare float @acoshf(float)\n"
      "declare x86_fp80 @acoshl(x86_fp80)\n"
      "declare x86_fp80 @acosl(x86_fp80)\n"
      "declare ptr @aligned_alloc(i64, i64)\n"
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
      "declare double @atof(ptr)\n"
      "declare i32 @atoi(ptr)\n"
      "declare i64 @atol(ptr)\n"
      "declare i64 @atoll(ptr)\n"
      "declare i32 @bcmp(ptr, ptr, i64)\n"
      "declare void @bcopy(ptr, ptr, i64)\n"
      "declare void @bzero(ptr, i64)\n"
      "declare ptr @calloc(i64, i64)\n"
      "declare double @cbrt(double)\n"
      "declare float @cbrtf(float)\n"
      "declare x86_fp80 @cbrtl(x86_fp80)\n"
      "declare double @ceil(double)\n"
      "declare float @ceilf(float)\n"
      "declare x86_fp80 @ceill(x86_fp80)\n"
      "declare i32 @chown(ptr, i32, i32)\n"
      "declare void @clearerr(ptr)\n"
      "declare double @copysign(double, double)\n"
      "declare float @copysignf(float, float)\n"
      "declare x86_fp80 @copysignl(x86_fp80, x86_fp80)\n"
      "declare double @cabs([2 x double])\n"
      "declare float @cabsf([2 x float])\n"
      "declare x86_fp80 @cabsl([2 x x86_fp80])\n"
      "declare double @cos(double)\n"
      "declare float @cosf(float)\n"
      "declare double @cosh(double)\n"
      "declare float @coshf(float)\n"
      "declare x86_fp80 @coshl(x86_fp80)\n"
      "declare x86_fp80 @cosl(x86_fp80)\n"
      "declare ptr @ctermid(ptr)\n"
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
      "declare i32 @fclose(ptr)\n"
      "declare i32 @feof(ptr)\n"
      "declare i32 @ferror(ptr)\n"
      "declare i32 @fflush(ptr)\n"
      "declare i32 @ffs(i32)\n"
      "declare i32 @ffsl(i64)\n"
      "declare i32 @ffsll(i64)\n"
      "declare i32 @fgetc(ptr)\n"
      "declare i32 @fgetc_unlocked(ptr)\n"
      "declare i32 @fgetpos(ptr, ptr)\n"
      "declare ptr @fgets(ptr, i32, ptr)\n"
      "declare ptr @fgets_unlocked(ptr, i32, ptr)\n"
      "declare i32 @fileno(ptr)\n"
      "declare void @flockfile(ptr)\n"
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
      "declare double @fmaximum_num(double, double)\n"
      "declare float @fmaximum_numf(float, float)\n"
      "declare x86_fp80 @fmaximum_numl(x86_fp80, x86_fp80)\n"
      "declare double @fminimum_num(double, double)\n"
      "declare float @fminimum_numf(float, float)\n"
      "declare x86_fp80 @fminimum_numl(x86_fp80, x86_fp80)\n"
      "declare double @fmod(double, double)\n"
      "declare float @fmodf(float, float)\n"
      "declare x86_fp80 @fmodl(x86_fp80, x86_fp80)\n"
      "declare i32 @fprintf(ptr, ptr, ...)\n"
      "declare i32 @fputc(i32, ptr)\n"
      "declare i32 @fputc_unlocked(i32, ptr)\n"
      "declare i64 @fread(ptr, i64, i64, ptr)\n"
      "declare i64 @fread_unlocked(ptr, i64, i64, ptr)\n"
      "declare void @free(ptr)\n"
      "declare double @frexp(double, ptr)\n"
      "declare float @frexpf(float, ptr)\n"
      "declare x86_fp80 @frexpl(x86_fp80, ptr)\n"
      "declare i32 @fscanf(ptr, ptr, ...)\n"
      "declare i32 @fseek(ptr, i64, i32)\n"
      "declare i32 @fseeko(ptr, i64, i32)\n"
      "declare i32 @fsetpos(ptr, ptr)\n"
      "declare i32 @fstatvfs(i32, ptr)\n"
      "declare i64 @ftell(ptr)\n"
      "declare i64 @ftello(ptr)\n"
      "declare i32 @ftrylockfile(ptr)\n"
      "declare void @funlockfile(ptr)\n"
      "declare i32 @getc(ptr)\n"
      "declare i32 @getc_unlocked(ptr)\n"
      "declare i32 @getchar()\n"
      "declare i32 @getchar_unlocked()\n"
      "declare ptr @getenv(ptr)\n"
      "declare i32 @getitimer(i32, ptr)\n"
      "declare i32 @getlogin_r(ptr, i64)\n"
      "declare ptr @getpwnam(ptr)\n"
      "declare ptr @gets(ptr)\n"
      "declare i32 @gettimeofday(ptr, ptr)\n"
      "declare double @hypot(double, double)\n"
      "declare float @hypotf(float, float)\n"
      "declare x86_fp80 @hypotl(x86_fp80, x86_fp80)\n"
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
      "declare i32 @ilogb(double)\n"
      "declare i32 @ilogbf(float)\n"
      "declare i32 @ilogbl(x86_fp80)\n"
      "declare double @logb(double)\n"
      "declare float @logbf(float)\n"
      "declare x86_fp80 @logbl(x86_fp80)\n"
      "declare float @logf(float)\n"
      "declare x86_fp80 @logl(x86_fp80)\n"
      "declare double @nextafter(double, double)\n"
      "declare float @nextafterf(float, float)\n"
      "declare x86_fp80 @nextafterl(x86_fp80, x86_fp80)\n"
      "declare double @nexttoward(double, x86_fp80)\n"
      "declare float @nexttowardf(float, x86_fp80)\n"
      "declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80)\n"
      "declare ptr @malloc(i64)\n"
      "declare ptr @memccpy(ptr, ptr, i32, i64)\n"
      "declare ptr @memchr(ptr, i32, i64)\n"
      "declare i32 @memcmp(ptr, ptr, i64)\n"
      "declare ptr @memcpy(ptr, ptr, i64)\n"
      "declare ptr @memmove(ptr, ptr, i64)\n"
      "declare ptr @memset(ptr, i32, i64)\n"
      "declare void @memset_pattern16(ptr, ptr, i64)\n"
      "declare void @memset_pattern4(ptr, ptr, i64)\n"
      "declare void @memset_pattern8(ptr, ptr, i64)\n"
      "declare i32 @mkdir(ptr, i16)\n"
      "declare double @modf(double, ptr)\n"
      "declare float @modff(float, ptr)\n"
      "declare x86_fp80 @modfl(x86_fp80, ptr)\n"
      "declare double @nan(ptr)\n"
      "declare float @nanf(ptr)\n"
      "declare x86_fp80 @nanl(ptr)\n"
      "declare double @nearbyint(double)\n"
      "declare float @nearbyintf(float)\n"
      "declare x86_fp80 @nearbyintl(x86_fp80)\n"
      "declare i32 @pclose(ptr)\n"
      "declare void @perror(ptr)\n"
      "declare i32 @posix_memalign(ptr, i64, i64)\n"
      "declare double @pow(double, double)\n"
      "declare float @powf(float, float)\n"
      "declare x86_fp80 @powl(x86_fp80, x86_fp80)\n"
      "declare double @erf(double)\n"
      "declare float @erff(float)\n"
      "declare x86_fp80 @erfl(x86_fp80)\n"
      "declare double @tgamma(double)\n"
      "declare float @tgammaf(float)\n"
      "declare x86_fp80 @tgammal(x86_fp80)\n"
      "declare i32 @printf(ptr, ...)\n"
      "declare i32 @putc(i32, ptr)\n"
      "declare i32 @putc_unlocked(i32, ptr)\n"
      "declare i32 @putchar(i32)\n"
      "declare i32 @putchar_unlocked(i32)\n"
      "declare i32 @puts(ptr)\n"
      "declare ptr @pvalloc(i64)\n"
      "declare void @qsort(ptr, i64, i64, ptr)\n"
      "declare i64 @readlink(ptr, ptr, i64)\n"
      "declare ptr @realloc(ptr, i64)\n"
      "declare ptr @reallocarray(ptr, i64, i64)\n"
      "declare ptr @reallocf(ptr, i64)\n"
      "declare double @remainder(double, double)\n"
      "declare float @remainderf(float, float)\n"
      "declare x86_fp80 @remainderl(x86_fp80, x86_fp80)\n"
      "declare i32 @remove(ptr)\n"
      "declare double @remquo(double, double, ptr)\n"
      "declare float @remquof(float, float, ptr)\n"
      "declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr)\n"
      "declare double @fdim(double, double)\n"
      "declare float @fdimf(float, float)\n"
      "declare x86_fp80 @fdiml(x86_fp80, x86_fp80)\n"
      "declare i32 @rename(ptr, ptr)\n"
      "declare void @rewind(ptr)\n"
      "declare double @rint(double)\n"
      "declare float @rintf(float)\n"
      "declare x86_fp80 @rintl(x86_fp80)\n"
      "declare i32 @rmdir(ptr)\n"
      "declare double @round(double)\n"
      "declare float @roundf(float)\n"
      "declare x86_fp80 @roundl(x86_fp80)\n"
      "declare double @roundeven(double)\n"
      "declare float @roundevenf(float)\n"
      "declare x86_fp80 @roundevenl(x86_fp80)\n"
      "declare double @scalbln(double, i64)\n"
      "declare float @scalblnf(float, i64)\n"
      "declare x86_fp80 @scalblnl(x86_fp80, i64)\n"
      "declare double @scalbn(double, i32)\n"
      "declare float @scalbnf(float, i32)\n"
      "declare x86_fp80 @scalbnl(x86_fp80, i32)\n"
      "declare i32 @scanf(ptr, ...)\n"
      "declare void @setbuf(ptr, ptr)\n"
      "declare i32 @setitimer(i32, ptr, ptr)\n"
      "declare i32 @setvbuf(ptr, ptr, i32, i64)\n"
      "declare double @sin(double)\n"
      "declare float @sinf(float)\n"
      "declare double @sinh(double)\n"
      "declare float @sinhf(float)\n"
      "declare x86_fp80 @sinhl(x86_fp80)\n"
      "declare x86_fp80 @sinl(x86_fp80)\n"
      "declare void @sincos(double, ptr, ptr)\n"
      "declare void @sincosf(float, ptr, ptr)\n"
      "declare void @sincosl(x86_fp80, ptr, ptr)\n"
      "declare i32 @snprintf(ptr, i64, ptr, ...)\n"
      "declare i32 @sprintf(ptr, ptr, ...)\n"
      "declare double @sqrt(double)\n"
      "declare float @sqrtf(float)\n"
      "declare x86_fp80 @sqrtl(x86_fp80)\n"
      "declare i32 @sscanf(ptr, ptr, ...)\n"
      "declare i32 @statvfs(ptr, ptr)\n"
      "declare ptr @stpcpy(ptr, ptr)\n"
      "declare ptr @stpncpy(ptr, ptr, i64)\n"
      "declare i32 @strcasecmp(ptr, ptr)\n"
      "declare ptr @strcat(ptr, ptr)\n"
      "declare ptr @strchr(ptr, i32)\n"
      "declare i32 @strcmp(ptr, ptr)\n"
      "declare i32 @strcoll(ptr, ptr)\n"
      "declare ptr @strcpy(ptr, ptr)\n"
      "declare i64 @strcspn(ptr, ptr)\n"
      "declare ptr @strdup(ptr)\n"
      "declare i64 @strlen(ptr)\n"
      "declare i32 @strncasecmp(ptr, ptr, i64)\n"
      "declare ptr @strncat(ptr, ptr, i64)\n"
      "declare i32 @strncmp(ptr, ptr, i64)\n"
      "declare ptr @strncpy(ptr, ptr, i64)\n"
      "declare ptr @strndup(ptr, i64)\n"
      "declare i64 @strnlen(ptr, i64)\n"
      "declare ptr @strpbrk(ptr, ptr)\n"
      "declare ptr @strrchr(ptr, i32)\n"
      "declare i64 @strspn(ptr, ptr)\n"
      "declare ptr @strstr(ptr, ptr)\n"
      "declare ptr @strtok(ptr, ptr)\n"
      "declare ptr @strtok_r(ptr, ptr, ptr)\n"
      "declare i64 @strtol(ptr, ptr, i32)\n"
      "declare i64 @strlcat(ptr, ptr, i64)\n"
      "declare i64 @strlcpy(ptr, ptr, i64)\n"
      "declare x86_fp80 @strtold(ptr, ptr)\n"
      "declare i64 @strtoll(ptr, ptr, i32)\n"
      "declare i64 @strtoul(ptr, ptr, i32)\n"
      "declare i64 @strtoull(ptr, ptr, i32)\n"
      "declare i64 @strxfrm(ptr, ptr, i64)\n"
      "declare double @tan(double)\n"
      "declare float @tanf(float)\n"
      "declare double @tanh(double)\n"
      "declare float @tanhf(float)\n"
      "declare x86_fp80 @tanhl(x86_fp80)\n"
      "declare x86_fp80 @tanl(x86_fp80)\n"
      "declare i64 @times(ptr)\n"
      "declare ptr @tmpfile()\n"
      "declare i32 @_Z7toasciii(i32)\n"
      "declare double @trunc(double)\n"
      "declare float @truncf(float)\n"
      "declare x86_fp80 @truncl(x86_fp80)\n"
      "declare i32 @uname(ptr)\n"
      "declare i32 @ungetc(i32, ptr)\n"
      "declare i32 @unlink(ptr)\n"
      "declare i32 @utime(ptr, ptr)\n"
      "declare i32 @utimes(ptr, ptr)\n"
      "declare ptr @valloc(i64)\n"
      "declare i32 @vfprintf(ptr, ptr, ptr)\n"
      "declare i32 @vfscanf(ptr, ptr, ptr)\n"
      "declare i32 @vprintf(ptr, ptr)\n"
      "declare i32 @vscanf(ptr, ptr)\n"
      "declare i32 @vsnprintf(ptr, i64, ptr, ptr)\n"
      "declare i32 @vsprintf(ptr, ptr, ptr)\n"
      "declare i32 @vsscanf(ptr, ptr, ptr)\n"
      "declare i64 @wcslen(ptr)\n"
      "declare i32 @fork()\n"
      "declare i32 @execl(ptr, ptr, ...)\n"
      "declare i32 @execle(ptr, ptr, ...)\n"
      "declare i32 @execlp(ptr, ptr, ...)\n"
      "declare i32 @execv(ptr, ptr)\n"
      "declare i32 @execvP(ptr, ptr, ptr)\n"
      "declare i32 @execve(ptr, ptr, ptr)\n"
      "declare i32 @execvp(ptr, ptr)\n"
      "declare i32 @execvpe(ptr, ptr, ptr)\n"

      // These functions were also extracted from the OS X headers, but they are
      // available with a special name on darwin.
      // This test uses the default TLI name instead.
      "declare i32 @chmod(ptr, i16)\n"
      "declare i32 @closedir(ptr)\n"
      "declare ptr @fdopen(i32, ptr)\n"
      "declare ptr @fopen(ptr, ptr)\n"
      "declare i32 @fputs(ptr, ptr)\n"
      "declare i32 @fputs_unlocked(ptr, ptr)\n"
      "declare i32 @fstat(i32, ptr)\n"
      "declare i64 @fwrite(ptr, i64, i64, ptr)\n"
      "declare i64 @fwrite_unlocked(ptr, i64, i64, ptr)\n"
      "declare i32 @lchown(ptr, i32, i32)\n"
      "declare i32 @lstat(ptr, ptr)\n"
      "declare i64 @mktime(ptr)\n"
      "declare i32 @open(ptr, i32, ...)\n"
      "declare ptr @opendir(ptr)\n"
      "declare ptr @popen(ptr, ptr)\n"
      "declare i64 @pread(i32, ptr, i64, i64)\n"
      "declare i64 @pwrite(i32, ptr, i64, i64)\n"
      "declare i64 @read(i32, ptr, i64)\n"
      "declare ptr @realpath(ptr, ptr)\n"
      "declare i32 @stat(ptr, ptr)\n"
      "declare double @strtod(ptr, ptr)\n"
      "declare float @strtof(ptr, ptr)\n"
      "declare i32 @system(ptr)\n"
      "declare i32 @unsetenv(ptr)\n"
      "declare i64 @write(i32, ptr, i64)\n"

      // These functions are available on Linux but not Darwin; they only differ
      // from their non-64 counterparts in the struct type.
      // Use the same prototype as the non-64 variant.
      "declare ptr @fopen64(ptr, ptr)\n"
      "declare i32 @fstat64(i32, ptr)\n"
      "declare i32 @fstatvfs64(i32, ptr)\n"
      "declare i32 @lstat64(ptr, ptr)\n"
      "declare i32 @open64(ptr, i32, ...)\n"
      "declare i32 @stat64(ptr, ptr)\n"
      "declare i32 @statvfs64(ptr, ptr)\n"
      "declare ptr @tmpfile64()\n"

      // These functions are also -64 variants, but do differ in the type of the
      // off_t (vs off64_t) parameter.  The non-64 variants declared above used
      // a 64-bit off_t, so, in practice, they are also equivalent.
      "declare i32 @fseeko64(ptr, i64, i32)\n"
      "declare i64 @ftello64(ptr)\n"

      "declare void @_ZdaPv(ptr)\n"
      "declare void @_ZdaPvRKSt9nothrow_t(ptr, ptr)\n"
      "declare void @_ZdaPvSt11align_val_t(ptr, i64)\n"
      "declare void @_ZdaPvSt11align_val_tRKSt9nothrow_t(ptr, i64, ptr)\n"
      "declare void @_ZdaPvj(ptr, i32)\n"
      "declare void @_ZdaPvjSt11align_val_t(ptr, i32, i32)\n"
      "declare void @_ZdaPvm(ptr, i64)\n"
      "declare void @_ZdaPvmSt11align_val_t(ptr, i64, i64)\n"
      "declare void @_ZdlPv(ptr)\n"
      "declare void @_ZdlPvRKSt9nothrow_t(ptr, ptr)\n"
      "declare void @_ZdlPvSt11align_val_t(ptr, i64)\n"
      "declare void @_ZdlPvSt11align_val_tRKSt9nothrow_t(ptr, i64, ptr)\n"
      "declare void @_ZdlPvj(ptr, i32)\n"
      "declare void @_ZdlPvjSt11align_val_t(ptr, i32, i32)\n"
      "declare void @_ZdlPvm(ptr, i64)\n"
      "declare void @_ZdlPvmSt11align_val_t(ptr, i64, i64)\n"
      "declare ptr @_Znaj(i32)\n"
      "declare ptr @_ZnajRKSt9nothrow_t(i32, ptr)\n"
      "declare ptr @_ZnajSt11align_val_t(i32, i32)\n"
      "declare ptr @_ZnajSt11align_val_tRKSt9nothrow_t(i32, i32, ptr)\n"
      "declare ptr @_Znam(i64)\n"
      "declare ptr @_Znam12__hot_cold_t(i64, i8)\n"
      "declare ptr @_ZnamRKSt9nothrow_t(i64, ptr)\n"
      "declare ptr @_ZnamRKSt9nothrow_t12__hot_cold_t(i64, ptr, i8)\n"
      "declare ptr @_ZnamSt11align_val_t(i64, i64)\n"
      "declare ptr @_ZnamSt11align_val_t12__hot_cold_t(i64, i64, i8)\n"
      "declare ptr @_ZnamSt11align_val_tRKSt9nothrow_t(i64, i64, ptr)\n"
      "declare ptr @_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64, i64, "
      "ptr, i8)\n"
      "declare ptr @_Znwj(i32)\n"
      "declare ptr @_ZnwjRKSt9nothrow_t(i32, ptr)\n"
      "declare ptr @_ZnwjSt11align_val_t(i32, i32)\n"
      "declare ptr @_ZnwjSt11align_val_tRKSt9nothrow_t(i32, i32, ptr)\n"
      "declare ptr @_Znwm(i64)\n"
      "declare ptr @_Znwm12__hot_cold_t(i64, i8)\n"
      "declare ptr @_ZnwmRKSt9nothrow_t(i64, ptr)\n"
      "declare ptr @_ZnwmRKSt9nothrow_t12__hot_cold_t(i64, ptr, i8)\n"
      "declare ptr @_ZnwmSt11align_val_t(i64, i64)\n"
      "declare ptr @_ZnwmSt11align_val_t12__hot_cold_t(i64, i64, i8)\n"
      "declare ptr @_ZnwmSt11align_val_tRKSt9nothrow_t(i64, i64, ptr)\n"
      "declare ptr @_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t(i64, i64, "
      "ptr, i8)\n"
      "declare {ptr, i64} @__size_returning_new(i64)\n"
      "declare {ptr, i64} @__size_returning_new_hot_cold(i64, i8)\n"
      "declare {ptr, i64} @__size_returning_new_aligned(i64, i64)\n"
      "declare {ptr, i64} @__size_returning_new_aligned_hot_cold(i64, i64, "
      "i8)\n"

      "declare void @\"??3@YAXPEAX@Z\"(ptr)\n"
      "declare void @\"??3@YAXPEAXAEBUnothrow_t@std@@@Z\"(ptr, ptr)\n"
      "declare void @\"??3@YAXPEAX_K@Z\"(ptr, i64)\n"
      "declare void @\"??_V@YAXPEAX@Z\"(ptr)\n"
      "declare void @\"??_V@YAXPEAXAEBUnothrow_t@std@@@Z\"(ptr, ptr)\n"
      "declare void @\"??_V@YAXPEAX_K@Z\"(ptr, i64)\n"
      "declare ptr @\"??2@YAPAXI@Z\"(i32)\n"
      "declare ptr @\"??2@YAPAXIABUnothrow_t@std@@@Z\"(i32, ptr)\n"
      "declare ptr @\"??2@YAPEAX_K@Z\"(i64)\n"
      "declare ptr @\"??2@YAPEAX_KAEBUnothrow_t@std@@@Z\"(i64, ptr)\n"
      "declare ptr @\"??_U@YAPAXI@Z\"(i32)\n"
      "declare ptr @\"??_U@YAPAXIABUnothrow_t@std@@@Z\"(i32, ptr)\n"
      "declare ptr @\"??_U@YAPEAX_K@Z\"(i64)\n"
      "declare ptr @\"??_U@YAPEAX_KAEBUnothrow_t@std@@@Z\"(i64, ptr)\n"

      "declare void @\"??3@YAXPAX@Z\"(ptr)\n"
      "declare void @\"??3@YAXPAXABUnothrow_t@std@@@Z\"(ptr, ptr)\n"
      "declare void @\"??3@YAXPAXI@Z\"(ptr, i32)\n"
      "declare void @\"??_V@YAXPAX@Z\"(ptr)\n"
      "declare void @\"??_V@YAXPAXABUnothrow_t@std@@@Z\"(ptr, ptr)\n"
      "declare void @\"??_V@YAXPAXI@Z\"(ptr, i32)\n"

      // These other functions were derived from the .def C declaration.
      "declare i32 @__cxa_atexit(ptr, ptr, ptr)\n"
      "declare void @__cxa_guard_abort(ptr)\n"
      "declare i32 @__cxa_guard_acquire(ptr)\n"
      "declare void @__cxa_guard_release(ptr)\n"
      "declare void @__cxa_throw(ptr, ptr, ptr)\n"

      "declare i32 @atexit(ptr)\n"

      "declare void @abort()\n"
      "declare void @exit(i32)\n"
      "declare void @_Exit(i32)\n"
      "declare void @_ZSt9terminatev()\n"

      "declare i32 @__nvvm_reflect(ptr)\n"

      "declare ptr @__memcpy_chk(ptr, ptr, i64, i64)\n"
      "declare ptr @__memmove_chk(ptr, ptr, i64, i64)\n"
      "declare ptr @__memset_chk(ptr, i32, i64, i64)\n"
      "declare ptr @__stpcpy_chk(ptr, ptr, i64)\n"
      "declare ptr @__stpncpy_chk(ptr, ptr, i64, i64)\n"
      "declare ptr @__strcpy_chk(ptr, ptr, i64)\n"
      "declare ptr @__strncpy_chk(ptr, ptr, i64, i64)\n"
      "declare ptr @__memccpy_chk(ptr, ptr, i32, i64, i64)\n"
      "declare ptr @__mempcpy_chk(ptr, ptr, i64, i64)\n"
      "declare i32 @__snprintf_chk(ptr, i64, i32, i64, ptr, ...)\n"
      "declare i32 @__sprintf_chk(ptr, i32, i64, ptr, ...)\n"
      "declare ptr @__strcat_chk(ptr, ptr, i64)\n"
      "declare i64 @__strlcat_chk(ptr, ptr, i64, i64)\n"
      "declare i64 @__strlen_chk(ptr, i64)\n"
      "declare ptr @__strncat_chk(ptr, ptr, i64, i64)\n"
      "declare i64 @__strlcpy_chk(ptr, ptr, i64, i64)\n"
      "declare i32 @__vsnprintf_chk(ptr, i64, i32, i64, ptr, ptr)\n"
      "declare i32 @__vsprintf_chk(ptr, i32, i64, ptr, ptr)\n"

      "declare ptr @memalign(i64, i64)\n"
      "declare ptr @mempcpy(ptr, ptr, i64)\n"
      "declare ptr @memrchr(ptr, i32, i64)\n"

      "declare void @__atomic_load(i64, ptr, ptr, i32)\n"
      "declare void @__atomic_store(i64, ptr, ptr, i32)\n"

      // These are similar to the FILE* fgetc/fputc.
      "declare i32 @_IO_getc(ptr)\n"
      "declare i32 @_IO_putc(i32, ptr)\n"

      "declare i32 @__isoc99_scanf(ptr, ...)\n"
      "declare i32 @__isoc99_sscanf(ptr, ptr, ...)\n"
      "declare ptr @__strdup(ptr)\n"
      "declare ptr @__strndup(ptr, i64)\n"
      "declare ptr @__strtok_r(ptr, ptr, ptr)\n"

      "declare double @__sqrt_finite(double)\n"
      "declare float @__sqrtf_finite(float)\n"
      "declare x86_fp80 @__sqrtl_finite(x86_fp80)\n"
      "declare double @exp10(double)\n"
      "declare float @exp10f(float)\n"
      "declare x86_fp80 @exp10l(x86_fp80)\n"

      // These printf variants have the same prototype as the non-'i' versions.
      "declare i32 @fiprintf(ptr, ptr, ...)\n"
      "declare i32 @iprintf(ptr, ...)\n"
      "declare i32 @siprintf(ptr, ptr, ...)\n"

      // __small_printf variants have the same prototype as the non-'i'
      // versions.
      "declare i32 @__small_fprintf(ptr, ptr, ...)\n"
      "declare i32 @__small_printf(ptr, ...)\n"
      "declare i32 @__small_sprintf(ptr, ptr, ...)\n"

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

      // These functions are aix vec allocation/free routines
      "declare ptr @vec_calloc(i64, i64)\n"
      "declare ptr @vec_malloc(i64)\n"
      "declare ptr @vec_realloc(ptr, i64)\n"
      "declare void @vec_free(ptr)\n"

      // These functions are OpenMP Offloading allocation / free routines
      "declare ptr @__kmpc_alloc_shared(i64)\n"
      "declare void @__kmpc_free_shared(ptr, i64)\n");

  for (unsigned FI = LibFunc::Begin_LibFunc; FI != LibFunc::End_LibFunc; ++FI) {
    LibFunc LF = (LibFunc)FI;
    // Make sure everything is available; we're not testing target defaults.
    TLII.setAvailable(LF);
    Function *F = M->getFunction(TLI.getName(LF));
    EXPECT_TRUE(isLibFunc(F, LF));
  }
}

namespace {

/// Creates TLI for AArch64 and uses it to get the LibFunc names for the given
/// Instruction opcode and Type.
class TLITestAarch64 : public ::testing::Test {
private:
  const Triple TargetTriple;

protected:
  LLVMContext Ctx;
  std::unique_ptr<TargetLibraryInfoImpl> TLII;
  std::unique_ptr<TargetLibraryInfo> TLI;

  /// Create TLI for AArch64
  TLITestAarch64() : TargetTriple(Triple("aarch64-unknown-linux-gnu")) {
    TLII = std::make_unique<TargetLibraryInfoImpl>(
        TargetLibraryInfoImpl(TargetTriple));
    TLI = std::make_unique<TargetLibraryInfo>(TargetLibraryInfo(*TLII));
  }

  /// Returns the TLI function name for the given \p Opcode and type \p Ty.
  StringRef getScalarName(unsigned int Opcode, Type *Ty) {
    LibFunc Func;
    if (!TLI->getLibFunc(Opcode, Ty, Func))
      return "";
    return TLI->getName(Func);
  }
};
} // end anonymous namespace

TEST_F(TLITestAarch64, TestFrem) {
  EXPECT_EQ(getScalarName(Instruction::FRem, Type::getDoubleTy(Ctx)), "fmod");
  EXPECT_EQ(getScalarName(Instruction::FRem, Type::getFloatTy(Ctx)), "fmodf");
}
