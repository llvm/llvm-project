; RUN: opt < %s -mtriple=x86_64-- -inferattrs -S | FileCheck -check-prefix=CHECK-UNKNOWN %s
; RUN: opt < %s -mtriple=x86_64-- -passes=inferattrs -S | FileCheck -check-prefix=CHECK-UNKNOWN %s
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.8.0 -inferattrs -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-DARWIN %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -inferattrs -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-LINUX %s
; RUN: opt < %s -mtriple=nvptx -inferattrs -S | FileCheck -check-prefix=CHECK-NVPTX %s

; operator new routines
declare i8* @_Znwj(i64)
; CHECK: declare noalias nonnull i8* @_Znwj(i64)
declare i8* @_Znwm(i64)
; CHECK: declare noalias nonnull i8* @_Znwm(i64)

declare i32 @__nvvm_reflect(i8*)
; CHECK-NVPTX: declare i32 @__nvvm_reflect(i8*) [[G0:#[0-9]+]]
; CHECK-NVPTX: attributes [[G0]] = { nounwind readnone }


; Check all the libc functions (thereby also exercising the prototype check).
; Note that it's OK to modify these as attributes might be missing. These checks
; reflect the currently inferred attributes.

; Use an opaque pointer type for all the (possibly opaque) structs.
%opaque = type opaque

; CHECK: declare double @__acos_finite(double)
declare double @__acos_finite(double)

; CHECK: declare float @__acosf_finite(float)
declare float @__acosf_finite(float)

; CHECK: declare double @__acosh_finite(double)
declare double @__acosh_finite(double)

; CHECK: declare float @__acoshf_finite(float)
declare float @__acoshf_finite(float)

; CHECK: declare x86_fp80 @__acoshl_finite(x86_fp80)
declare x86_fp80 @__acoshl_finite(x86_fp80)

; CHECK: declare x86_fp80 @__acosl_finite(x86_fp80)
declare x86_fp80 @__acosl_finite(x86_fp80)

; CHECK: declare double @__asin_finite(double)
declare double @__asin_finite(double)

; CHECK: declare float @__asinf_finite(float)
declare float @__asinf_finite(float)

; CHECK: declare x86_fp80 @__asinl_finite(x86_fp80)
declare x86_fp80 @__asinl_finite(x86_fp80)

; CHECK: declare double @__atan2_finite(double, double)
declare double @__atan2_finite(double, double)

; CHECK: declare float @__atan2f_finite(float, float)
declare float @__atan2f_finite(float, float)

; CHECK: declare x86_fp80 @__atan2l_finite(x86_fp80, x86_fp80)
declare x86_fp80 @__atan2l_finite(x86_fp80, x86_fp80)

; CHECK: declare double @__atanh_finite(double)
declare double @__atanh_finite(double)

; CHECK: declare float @__atanhf_finite(float)
declare float @__atanhf_finite(float)

; CHECK: declare x86_fp80 @__atanhl_finite(x86_fp80)
declare x86_fp80 @__atanhl_finite(x86_fp80)

; CHECK: declare double @__cosh_finite(double)
declare double @__cosh_finite(double)

; CHECK: declare float @__coshf_finite(float)
declare float @__coshf_finite(float)

; CHECK: declare x86_fp80 @__coshl_finite(x86_fp80)
declare x86_fp80 @__coshl_finite(x86_fp80)

; CHECK: declare double @__cospi(double)
declare double @__cospi(double)

; CHECK: declare float @__cospif(float)
declare float @__cospif(float)

; CHECK: declare double @__exp10_finite(double)
declare double @__exp10_finite(double)

; CHECK: declare float @__exp10f_finite(float)
declare float @__exp10f_finite(float)

; CHECK: declare x86_fp80 @__exp10l_finite(x86_fp80)
declare x86_fp80 @__exp10l_finite(x86_fp80)

; CHECK: declare double @__exp2_finite(double)
declare double @__exp2_finite(double)

; CHECK: declare float @__exp2f_finite(float)
declare float @__exp2f_finite(float)

; CHECK: declare x86_fp80 @__exp2l_finite(x86_fp80)
declare x86_fp80 @__exp2l_finite(x86_fp80)

; CHECK: declare double @__exp_finite(double)
declare double @__exp_finite(double)

; CHECK: declare float @__expf_finite(float)
declare float @__expf_finite(float)

; CHECK: declare x86_fp80 @__expl_finite(x86_fp80)
declare x86_fp80 @__expl_finite(x86_fp80)

; CHECK: declare double @__log10_finite(double)
declare double @__log10_finite(double)

; CHECK: declare float @__log10f_finite(float)
declare float @__log10f_finite(float)

; CHECK: declare x86_fp80 @__log10l_finite(x86_fp80)
declare x86_fp80 @__log10l_finite(x86_fp80)

; CHECK: declare double @__log2_finite(double)
declare double @__log2_finite(double)

; CHECK: declare float @__log2f_finite(float)
declare float @__log2f_finite(float)

; CHECK: declare x86_fp80 @__log2l_finite(x86_fp80)
declare x86_fp80 @__log2l_finite(x86_fp80)

; CHECK: declare double @__log_finite(double)
declare double @__log_finite(double)

; CHECK: declare float @__logf_finite(float)
declare float @__logf_finite(float)

; CHECK: declare x86_fp80 @__logl_finite(x86_fp80)
declare x86_fp80 @__logl_finite(x86_fp80)

; CHECK: declare double @__pow_finite(double, double)
declare double @__pow_finite(double, double)

; CHECK: declare float @__powf_finite(float, float)
declare float @__powf_finite(float, float)

; CHECK: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80)
declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80)

; CHECK: declare double @__sinh_finite(double)
declare double @__sinh_finite(double)

; CHECK: declare float @__sinhf_finite(float)
declare float @__sinhf_finite(float)

; CHECK: declare x86_fp80 @__sinhl_finite(x86_fp80)
declare x86_fp80 @__sinhl_finite(x86_fp80)

; CHECK: declare double @__sinpi(double)
declare double @__sinpi(double)

; CHECK: declare float @__sinpif(float)
declare float @__sinpif(float)

; CHECK: declare i32 @abs(i32)
declare i32 @abs(i32)

; CHECK: declare i32 @access(i8* nocapture readonly, i32) [[G0:#[0-9]+]]
declare i32 @access(i8*, i32)

; CHECK: declare double @acos(double)
declare double @acos(double)

; CHECK: declare float @acosf(float)
declare float @acosf(float)

; CHECK: declare double @acosh(double)
declare double @acosh(double)

; CHECK: declare float @acoshf(float)
declare float @acoshf(float)

; CHECK: declare x86_fp80 @acoshl(x86_fp80)
declare x86_fp80 @acoshl(x86_fp80)

; CHECK: declare x86_fp80 @acosl(x86_fp80)
declare x86_fp80 @acosl(x86_fp80)

; CHECK: declare double @asin(double)
declare double @asin(double)

; CHECK: declare float @asinf(float)
declare float @asinf(float)

; CHECK: declare double @asinh(double)
declare double @asinh(double)

; CHECK: declare float @asinhf(float)
declare float @asinhf(float)

; CHECK: declare x86_fp80 @asinhl(x86_fp80)
declare x86_fp80 @asinhl(x86_fp80)

; CHECK: declare x86_fp80 @asinl(x86_fp80)
declare x86_fp80 @asinl(x86_fp80)

; CHECK: declare double @atan(double)
declare double @atan(double)

; CHECK: declare double @atan2(double, double)
declare double @atan2(double, double)

; CHECK: declare float @atan2f(float, float)
declare float @atan2f(float, float)

; CHECK: declare x86_fp80 @atan2l(x86_fp80, x86_fp80)
declare x86_fp80 @atan2l(x86_fp80, x86_fp80)

; CHECK: declare float @atanf(float)
declare float @atanf(float)

; CHECK: declare double @atanh(double)
declare double @atanh(double)

; CHECK: declare float @atanhf(float)
declare float @atanhf(float)

; CHECK: declare x86_fp80 @atanhl(x86_fp80)
declare x86_fp80 @atanhl(x86_fp80)

; CHECK: declare x86_fp80 @atanl(x86_fp80)
declare x86_fp80 @atanl(x86_fp80)

; CHECK: declare double @atof(i8* nocapture) [[G1:#[0-9]+]]
declare double @atof(i8*)

; CHECK: declare i32 @atoi(i8* nocapture) [[G1]]
declare i32 @atoi(i8*)

; CHECK: declare i64 @atol(i8* nocapture) [[G1]]
declare i64 @atol(i8*)

; CHECK: declare i64 @atoll(i8* nocapture) [[G1]]
declare i64 @atoll(i8*)

; CHECK-DARWIN: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G1]]
; CHECK-LINUX: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G1]]
; CHECK-UNKNOWN-NOT: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G1]]
; CHECK-NVPTX-NOT: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G1]]
declare i32 @bcmp(i8*, i8*, i64)

; CHECK: declare void @bcopy(i8* nocapture readonly, i8* nocapture, i64) [[G0]]
declare void @bcopy(i8*, i8*, i64)

; CHECK: declare void @bzero(i8* nocapture, i64) [[G0]]
declare void @bzero(i8*, i64)

; CHECK: declare noalias i8* @calloc(i64, i64) [[G0]]
declare i8* @calloc(i64, i64)

; CHECK: declare double @cbrt(double)
declare double @cbrt(double)

; CHECK: declare float @cbrtf(float)
declare float @cbrtf(float)

; CHECK: declare x86_fp80 @cbrtl(x86_fp80)
declare x86_fp80 @cbrtl(x86_fp80)

; CHECK: declare double @ceil(double)
declare double @ceil(double)

; CHECK: declare float @ceilf(float)
declare float @ceilf(float)

; CHECK: declare x86_fp80 @ceill(x86_fp80)
declare x86_fp80 @ceill(x86_fp80)

; CHECK: declare i32 @chmod(i8* nocapture readonly, i16 zeroext) [[G0]]
declare i32 @chmod(i8*, i16 zeroext)

; CHECK: declare i32 @chown(i8* nocapture readonly, i32, i32) [[G0]]
declare i32 @chown(i8*, i32, i32)

; CHECK: declare void @clearerr(%opaque* nocapture) [[G0]]
declare void @clearerr(%opaque*)

; CHECK: declare i32 @closedir(%opaque* nocapture) [[G0]]
declare i32 @closedir(%opaque*)

; CHECK: declare double @copysign(double, double)
declare double @copysign(double, double)

; CHECK: declare float @copysignf(float, float)
declare float @copysignf(float, float)

; CHECK: declare x86_fp80 @copysignl(x86_fp80, x86_fp80)
declare x86_fp80 @copysignl(x86_fp80, x86_fp80)

; CHECK: declare double @cos(double)
declare double @cos(double)

; CHECK: declare float @cosf(float)
declare float @cosf(float)

; CHECK: declare double @cosh(double)
declare double @cosh(double)

; CHECK: declare float @coshf(float)
declare float @coshf(float)

; CHECK: declare x86_fp80 @coshl(x86_fp80)
declare x86_fp80 @coshl(x86_fp80)

; CHECK: declare x86_fp80 @cosl(x86_fp80)
declare x86_fp80 @cosl(x86_fp80)

; CHECK: declare i8* @ctermid(i8* nocapture) [[G0]]
declare i8* @ctermid(i8*)

; CHECK: declare double @exp(double)
declare double @exp(double)

; CHECK: declare double @exp2(double)
declare double @exp2(double)

; CHECK: declare float @exp2f(float)
declare float @exp2f(float)

; CHECK: declare x86_fp80 @exp2l(x86_fp80)
declare x86_fp80 @exp2l(x86_fp80)

; CHECK: declare float @expf(float)
declare float @expf(float)

; CHECK: declare x86_fp80 @expl(x86_fp80)
declare x86_fp80 @expl(x86_fp80)

; CHECK: declare double @expm1(double)
declare double @expm1(double)

; CHECK: declare float @expm1f(float)
declare float @expm1f(float)

; CHECK: declare x86_fp80 @expm1l(x86_fp80)
declare x86_fp80 @expm1l(x86_fp80)

; CHECK: declare double @fabs(double)
declare double @fabs(double)

; CHECK: declare float @fabsf(float)
declare float @fabsf(float)

; CHECK: declare x86_fp80 @fabsl(x86_fp80)
declare x86_fp80 @fabsl(x86_fp80)

; CHECK: declare i32 @fclose(%opaque* nocapture) [[G0]]
declare i32 @fclose(%opaque*)

; CHECK: declare noalias %opaque* @fdopen(i32, i8* nocapture readonly) [[G0]]
declare %opaque* @fdopen(i32, i8*)

; CHECK: declare i32 @feof(%opaque* nocapture) [[G0]]
declare i32 @feof(%opaque*)

; CHECK: declare i32 @ferror(%opaque* nocapture) [[G1]]
declare i32 @ferror(%opaque*)

; CHECK: declare i32 @fflush(%opaque* nocapture) [[G0]]
declare i32 @fflush(%opaque*)

; CHECK: declare i32 @ffs(i32)
declare i32 @ffs(i32)

; CHECK: declare i32 @ffsl(i64)
declare i32 @ffsl(i64)

; CHECK: declare i32 @ffsll(i64)
declare i32 @ffsll(i64)

; CHECK: declare i32 @fgetc(%opaque* nocapture) [[G0]]
declare i32 @fgetc(%opaque*)

; CHECK: declare i32 @fgetpos(%opaque* nocapture, i64* nocapture) [[G0]]
declare i32 @fgetpos(%opaque*, i64*)

; CHECK: declare i8* @fgets(i8*, i32, %opaque* nocapture) [[G0]]
declare i8* @fgets(i8*, i32, %opaque*)

; CHECK: declare i32 @fileno(%opaque* nocapture) [[G0]]
declare i32 @fileno(%opaque*)

; CHECK: declare void @flockfile(%opaque* nocapture) [[G0]]
declare void @flockfile(%opaque*)

; CHECK: declare double @floor(double)
declare double @floor(double)

; CHECK: declare float @floorf(float)
declare float @floorf(float)

; CHECK: declare x86_fp80 @floorl(x86_fp80)
declare x86_fp80 @floorl(x86_fp80)

; CHECK: declare i32 @fls(i32)
declare i32 @fls(i32)

; CHECK: declare i32 @flsl(i64)
declare i32 @flsl(i64)

; CHECK: declare i32 @flsll(i64)
declare i32 @flsll(i64)

; CHECK: declare double @fmax(double, double)
declare double @fmax(double, double)

; CHECK: declare float @fmaxf(float, float)
declare float @fmaxf(float, float)

; CHECK: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)
declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)

; CHECK: declare double @fmin(double, double)
declare double @fmin(double, double)

; CHECK: declare float @fminf(float, float)
declare float @fminf(float, float)

; CHECK: declare x86_fp80 @fminl(x86_fp80, x86_fp80)
declare x86_fp80 @fminl(x86_fp80, x86_fp80)

; CHECK: declare double @fmod(double, double)
declare double @fmod(double, double)

; CHECK: declare float @fmodf(float, float)
declare float @fmodf(float, float)

; CHECK: declare x86_fp80 @fmodl(x86_fp80, x86_fp80)
declare x86_fp80 @fmodl(x86_fp80, x86_fp80)

; CHECK: declare noalias %opaque* @fopen(i8* nocapture readonly, i8* nocapture readonly) [[G0]]
declare %opaque* @fopen(i8*, i8*)

; CHECK: declare i32 @fprintf(%opaque* nocapture, i8* nocapture readonly, ...) [[G0]]
declare i32 @fprintf(%opaque*, i8*, ...)

; CHECK: declare i32 @fputc(i32, %opaque* nocapture) [[G0]]
declare i32 @fputc(i32, %opaque*)

; CHECK: declare i32 @fputs(i8* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @fputs(i8*, %opaque*)

; CHECK: declare i64 @fread(i8* nocapture, i64, i64, %opaque* nocapture) [[G0]]
declare i64 @fread(i8*, i64, i64, %opaque*)

; CHECK: declare void @free(i8* nocapture) [[G0]]
declare void @free(i8*)

; CHECK: declare double @frexp(double, i32* nocapture) [[G0]]
declare double @frexp(double, i32*)

; CHECK: declare float @frexpf(float, i32* nocapture) [[G0]]
declare float @frexpf(float, i32*)

; CHECK: declare x86_fp80 @frexpl(x86_fp80, i32* nocapture) [[G0]]
declare x86_fp80 @frexpl(x86_fp80, i32*)

; CHECK: declare i32 @fscanf(%opaque* nocapture, i8* nocapture readonly, ...) [[G0]]
declare i32 @fscanf(%opaque*, i8*, ...)

; CHECK: declare i32 @fseek(%opaque* nocapture, i64, i32) [[G0]]
declare i32 @fseek(%opaque*, i64, i32)

; CHECK: declare i32 @fseeko(%opaque* nocapture, i64, i32) [[G0]]
declare i32 @fseeko(%opaque*, i64, i32)

; CHECK-LINUX: declare i32 @fseeko64(%opaque* nocapture, i64, i32) [[G0]]
declare i32 @fseeko64(%opaque*, i64, i32)

; CHECK: declare i32 @fsetpos(%opaque* nocapture, i64*) [[G0]]
declare i32 @fsetpos(%opaque*, i64*)

; CHECK: declare i32 @fstat(i32, %opaque* nocapture) [[G0]]
declare i32 @fstat(i32, %opaque*)

; CHECK-LINUX: declare i32 @fstat64(i32, %opaque* nocapture) [[G0]]
declare i32 @fstat64(i32, %opaque*)

; CHECK: declare i32 @fstatvfs(i32, %opaque* nocapture) [[G0]]
declare i32 @fstatvfs(i32, %opaque*)

; CHECK-LINUX: declare i32 @fstatvfs64(i32, %opaque* nocapture) [[G0]]
declare i32 @fstatvfs64(i32, %opaque*)

; CHECK: declare i64 @ftell(%opaque* nocapture) [[G0]]
declare i64 @ftell(%opaque*)

; CHECK: declare i64 @ftello(%opaque* nocapture) [[G0]]
declare i64 @ftello(%opaque*)

; CHECK-LINUX: declare i64 @ftello64(%opaque* nocapture) [[G0]]
declare i64 @ftello64(%opaque*)

; CHECK: declare i32 @ftrylockfile(%opaque* nocapture) [[G0]]
declare i32 @ftrylockfile(%opaque*)

; CHECK: declare void @funlockfile(%opaque* nocapture) [[G0]]
declare void @funlockfile(%opaque*)

; CHECK: declare i64 @fwrite(i8* nocapture, i64, i64, %opaque* nocapture) [[G0]]
declare i64 @fwrite(i8*, i64, i64, %opaque*)

; CHECK: declare i32 @getc(%opaque* nocapture) [[G0]]
declare i32 @getc(%opaque*)

; CHECK: declare i32 @getc_unlocked(%opaque* nocapture) [[G0]]
declare i32 @getc_unlocked(%opaque*)

; CHECK: declare i32 @getchar()
declare i32 @getchar()

; CHECK: declare i32 @getchar_unlocked()
declare i32 @getchar_unlocked()

; CHECK: declare i8* @getenv(i8* nocapture) [[G1]]
declare i8* @getenv(i8*)

; CHECK: declare i32 @getitimer(i32, %opaque* nocapture) [[G0]]
declare i32 @getitimer(i32, %opaque*)

; CHECK: declare i32 @getlogin_r(i8* nocapture, i64) [[G0]]
declare i32 @getlogin_r(i8*, i64)

; CHECK: declare %opaque* @getpwnam(i8* nocapture readonly) [[G0]]
declare %opaque* @getpwnam(i8*)

; CHECK: declare i8* @gets(i8*)
declare i8* @gets(i8*)

; CHECK: declare i32 @gettimeofday(%opaque* nocapture, i8* nocapture) [[G0]]
declare i32 @gettimeofday(%opaque*, i8*)

; CHECK: declare i32 @isascii(i32)
declare i32 @isascii(i32)

; CHECK: declare i32 @isdigit(i32)
declare i32 @isdigit(i32)

; CHECK: declare i64 @labs(i64)
declare i64 @labs(i64)

; CHECK: declare i32 @lchown(i8* nocapture readonly, i32, i32) [[G0]]
declare i32 @lchown(i8*, i32, i32)

; CHECK: declare double @ldexp(double, i32)
declare double @ldexp(double, i32)

; CHECK: declare float @ldexpf(float, i32)
declare float @ldexpf(float, i32)

; CHECK: declare x86_fp80 @ldexpl(x86_fp80, i32)
declare x86_fp80 @ldexpl(x86_fp80, i32)

; CHECK: declare i64 @llabs(i64)
declare i64 @llabs(i64)

; CHECK: declare double @log(double)
declare double @log(double)

; CHECK: declare double @log10(double)
declare double @log10(double)

; CHECK: declare float @log10f(float)
declare float @log10f(float)

; CHECK: declare x86_fp80 @log10l(x86_fp80)
declare x86_fp80 @log10l(x86_fp80)

; CHECK: declare double @log1p(double)
declare double @log1p(double)

; CHECK: declare float @log1pf(float)
declare float @log1pf(float)

; CHECK: declare x86_fp80 @log1pl(x86_fp80)
declare x86_fp80 @log1pl(x86_fp80)

; CHECK: declare double @log2(double)
declare double @log2(double)

; CHECK: declare float @log2f(float)
declare float @log2f(float)

; CHECK: declare x86_fp80 @log2l(x86_fp80)
declare x86_fp80 @log2l(x86_fp80)

; CHECK: declare double @logb(double)
declare double @logb(double)

; CHECK: declare float @logbf(float)
declare float @logbf(float)

; CHECK: declare x86_fp80 @logbl(x86_fp80)
declare x86_fp80 @logbl(x86_fp80)

; CHECK: declare float @logf(float)
declare float @logf(float)

; CHECK: declare x86_fp80 @logl(x86_fp80)
declare x86_fp80 @logl(x86_fp80)

; CHECK: declare i32 @lstat(i8* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @lstat(i8*, %opaque*)

; CHECK-LINUX: declare i32 @lstat64(i8* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @lstat64(i8*, %opaque*)

; CHECK: declare noalias i8* @malloc(i64) [[G0]]
declare i8* @malloc(i64)

; CHECK-LINUX: declare noalias i8* @memalign(i64, i64)
declare i8* @memalign(i64, i64)

; CHECK: declare i8* @memccpy(i8*, i8* nocapture readonly, i32, i64) [[G0]]
declare i8* @memccpy(i8*, i8*, i32, i64)

; CHECK: declare i8* @memchr(i8*, i32, i64) [[G1]]
declare i8* @memchr(i8*, i32, i64)

; CHECK: declare i32 @memcmp(i8* nocapture, i8* nocapture, i64) [[G1]]
declare i32 @memcmp(i8*, i8*, i64)

; CHECK: declare i8* @memcpy(i8* returned, i8* nocapture readonly, i64) [[G0]]
declare i8* @memcpy(i8*, i8*, i64)

; CHECK: declare i8* @mempcpy(i8*, i8* nocapture readonly, i64) [[G0]]
declare i8* @mempcpy(i8*, i8*, i64)

; CHECK: declare i8* @memmove(i8* returned, i8* nocapture readonly, i64) [[G0]]
declare i8* @memmove(i8*, i8*, i64)

; CHECK: declare i8* @memset(i8*, i32, i64)
declare i8* @memset(i8*, i32, i64)

; CHECK: declare i32 @mkdir(i8* nocapture readonly, i16 zeroext) [[G0]]
declare i32 @mkdir(i8*, i16 zeroext)

; CHECK: declare i64 @mktime(%opaque* nocapture) [[G0]]
declare i64 @mktime(%opaque*)

; CHECK: declare double @modf(double, double* nocapture) [[G0]]
declare double @modf(double, double*)

; CHECK: declare float @modff(float, float* nocapture) [[G0]]
declare float @modff(float, float*)

; CHECK: declare x86_fp80 @modfl(x86_fp80, x86_fp80* nocapture) [[G0]]
declare x86_fp80 @modfl(x86_fp80, x86_fp80*)

; CHECK: declare double @nearbyint(double)
declare double @nearbyint(double)

; CHECK: declare float @nearbyintf(float)
declare float @nearbyintf(float)

; CHECK: declare x86_fp80 @nearbyintl(x86_fp80)
declare x86_fp80 @nearbyintl(x86_fp80)

; CHECK: declare i32 @open(i8* nocapture readonly, i32, ...)
declare i32 @open(i8*, i32, ...)

; CHECK-LINUX: declare i32 @open64(i8* nocapture readonly, i32, ...)
declare i32 @open64(i8*, i32, ...)

; CHECK: declare noalias %opaque* @opendir(i8* nocapture readonly) [[G0]]
declare %opaque* @opendir(i8*)

; CHECK: declare i32 @pclose(%opaque* nocapture) [[G0]]
declare i32 @pclose(%opaque*)

; CHECK: declare void @perror(i8* nocapture readonly) [[G0]]
declare void @perror(i8*)

; CHECK: declare noalias %opaque* @popen(i8* nocapture readonly, i8* nocapture readonly) [[G0]]
declare %opaque* @popen(i8*, i8*)

; CHECK: declare i32 @posix_memalign(i8**, i64, i64)
declare i32 @posix_memalign(i8**, i64, i64)

; CHECK: declare double @pow(double, double)
declare double @pow(double, double)

; CHECK: declare float @powf(float, float)
declare float @powf(float, float)

; CHECK: declare x86_fp80 @powl(x86_fp80, x86_fp80)
declare x86_fp80 @powl(x86_fp80, x86_fp80)

; CHECK: declare i64 @pread(i32, i8* nocapture, i64, i64)
declare i64 @pread(i32, i8*, i64, i64)

; CHECK: declare i32 @printf(i8* nocapture readonly, ...) [[G0]]
declare i32 @printf(i8*, ...)

; CHECK: declare i32 @putc(i32, %opaque* nocapture) [[G0]]
declare i32 @putc(i32, %opaque*)

; CHECK: declare i32 @putchar(i32)
declare i32 @putchar(i32)

; CHECK: declare i32 @putchar_unlocked(i32)
declare i32 @putchar_unlocked(i32)

; CHECK: declare i32 @puts(i8* nocapture readonly) [[G0]]
declare i32 @puts(i8*)

; CHECK: declare i64 @pwrite(i32, i8* nocapture readonly, i64, i64)
declare i64 @pwrite(i32, i8*, i64, i64)

; CHECK: declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)* nocapture)
declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)*)

; CHECK: declare i64 @read(i32, i8* nocapture, i64)
declare i64 @read(i32, i8*, i64)

; CHECK: declare i64 @readlink(i8* nocapture readonly, i8* nocapture, i64) [[G0]]
declare i64 @readlink(i8*, i8*, i64)

; CHECK: declare noalias i8* @realloc(i8* nocapture, i64) [[G0]]
declare i8* @realloc(i8*, i64)

; CHECK: declare i8* @reallocf(i8*, i64)
declare i8* @reallocf(i8*, i64)

; CHECK: declare i8* @realpath(i8* nocapture readonly, i8*)
declare i8* @realpath(i8*, i8*)

; CHECK: declare i32 @remove(i8* nocapture readonly) [[G0]]
declare i32 @remove(i8*)

; CHECK: declare i32 @rename(i8* nocapture readonly, i8* nocapture readonly) [[G0]]
declare i32 @rename(i8*, i8*)

; CHECK: declare void @rewind(%opaque* nocapture) [[G0]]
declare void @rewind(%opaque*)

; CHECK: declare double @rint(double)
declare double @rint(double)

; CHECK: declare float @rintf(float)
declare float @rintf(float)

; CHECK: declare x86_fp80 @rintl(x86_fp80)
declare x86_fp80 @rintl(x86_fp80)

; CHECK: declare i32 @rmdir(i8* nocapture readonly) [[G0]]
declare i32 @rmdir(i8*)

; CHECK: declare double @round(double)
declare double @round(double)

; CHECK: declare float @roundf(float)
declare float @roundf(float)

; CHECK: declare x86_fp80 @roundl(x86_fp80)
declare x86_fp80 @roundl(x86_fp80)

; CHECK: declare i32 @scanf(i8* nocapture readonly, ...) [[G0]]
declare i32 @scanf(i8*, ...)

; CHECK: declare void @setbuf(%opaque* nocapture, i8*) [[G0]]
declare void @setbuf(%opaque*, i8*)

; CHECK: declare i32 @setitimer(i32, %opaque* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @setitimer(i32, %opaque*, %opaque*)

; CHECK: declare i32 @setvbuf(%opaque* nocapture, i8*, i32, i64) [[G0]]
declare i32 @setvbuf(%opaque*, i8*, i32, i64)

; CHECK: declare double @sin(double)
declare double @sin(double)

; CHECK: declare float @sinf(float)
declare float @sinf(float)

; CHECK: declare double @sinh(double)
declare double @sinh(double)

; CHECK: declare float @sinhf(float)
declare float @sinhf(float)

; CHECK: declare x86_fp80 @sinhl(x86_fp80)
declare x86_fp80 @sinhl(x86_fp80)

; CHECK: declare x86_fp80 @sinl(x86_fp80)
declare x86_fp80 @sinl(x86_fp80)

; CHECK: declare i32 @snprintf(i8* nocapture, i64, i8* nocapture readonly, ...) [[G0]]
declare i32 @snprintf(i8*, i64, i8*, ...)

; CHECK: declare i32 @sprintf(i8* nocapture, i8* nocapture readonly, ...) [[G0]]
declare i32 @sprintf(i8*, i8*, ...)

; CHECK: declare double @sqrt(double)
declare double @sqrt(double)

; CHECK: declare float @sqrtf(float)
declare float @sqrtf(float)

; CHECK: declare x86_fp80 @sqrtl(x86_fp80)
declare x86_fp80 @sqrtl(x86_fp80)

; CHECK: declare i32 @sscanf(i8* nocapture readonly, i8* nocapture readonly, ...) [[G0]]
declare i32 @sscanf(i8*, i8*, ...)

; CHECK: declare i32 @stat(i8* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @stat(i8*, %opaque*)

; CHECK-LINUX: declare i32 @stat64(i8* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @stat64(i8*, %opaque*)

; CHECK: declare i32 @statvfs(i8* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @statvfs(i8*, %opaque*)

; CHECK-LINUX: declare i32 @statvfs64(i8* nocapture readonly, %opaque* nocapture) [[G0]]
declare i32 @statvfs64(i8*, %opaque*)

; CHECK: declare i8* @stpcpy(i8*, i8* nocapture readonly) [[G0]]
declare i8* @stpcpy(i8*, i8*)

; CHECK: declare i8* @stpncpy(i8*, i8* nocapture readonly, i64) [[G0]]
declare i8* @stpncpy(i8*, i8*, i64)

; CHECK: declare i32 @strcasecmp(i8* nocapture, i8* nocapture) [[G1]]
declare i32 @strcasecmp(i8*, i8*)

; CHECK: declare i8* @strcat(i8* returned, i8* nocapture readonly) [[G0]]
declare i8* @strcat(i8*, i8*)

; CHECK: declare i8* @strchr(i8*, i32) [[G1]]
declare i8* @strchr(i8*, i32)

; CHECK: declare i32 @strcmp(i8* nocapture, i8* nocapture) [[G1]]
declare i32 @strcmp(i8*, i8*)

; CHECK: declare i32 @strcoll(i8* nocapture, i8* nocapture) [[G1]]
declare i32 @strcoll(i8*, i8*)

; CHECK: declare i8* @strcpy(i8* returned, i8* nocapture readonly) [[G0]]
declare i8* @strcpy(i8*, i8*)

; CHECK: declare i64 @strcspn(i8* nocapture, i8* nocapture) [[G1]]
declare i64 @strcspn(i8*, i8*)

; CHECK: declare noalias i8* @strdup(i8* nocapture readonly) [[G0]]
declare i8* @strdup(i8*)

; CHECK: declare i64 @strlen(i8* nocapture) [[G2:#[0-9]+]]
declare i64 @strlen(i8*)

; CHECK: declare i32 @strncasecmp(i8* nocapture, i8* nocapture, i64) [[G1]]
declare i32 @strncasecmp(i8*, i8*, i64)

; CHECK: declare i8* @strncat(i8* returned, i8* nocapture readonly, i64) [[G0]]
declare i8* @strncat(i8*, i8*, i64)

; CHECK: declare i32 @strncmp(i8* nocapture, i8* nocapture, i64) [[G1]]
declare i32 @strncmp(i8*, i8*, i64)

; CHECK: declare i8* @strncpy(i8* returned, i8* nocapture readonly, i64) [[G0]]
declare i8* @strncpy(i8*, i8*, i64)

; CHECK: declare noalias i8* @strndup(i8* nocapture readonly, i64) [[G0]]
declare i8* @strndup(i8*, i64)

; CHECK: declare i64 @strnlen(i8*, i64)
declare i64 @strnlen(i8*, i64)

; CHECK: declare i8* @strpbrk(i8*, i8* nocapture) [[G1]]
declare i8* @strpbrk(i8*, i8*)

; CHECK: declare i8* @strrchr(i8*, i32) [[G1]]
declare i8* @strrchr(i8*, i32)

; CHECK: declare i64 @strspn(i8* nocapture, i8* nocapture) [[G1]]
declare i64 @strspn(i8*, i8*)

; CHECK: declare i8* @strstr(i8*, i8* nocapture) [[G1]]
declare i8* @strstr(i8*, i8*)

; CHECK: declare double @strtod(i8* readonly, i8** nocapture) [[G0]]
declare double @strtod(i8*, i8**)

; CHECK: declare float @strtof(i8* readonly, i8** nocapture) [[G0]]
declare float @strtof(i8*, i8**)

; CHECK: declare i8* @strtok(i8*, i8* nocapture readonly) [[G0]]
declare i8* @strtok(i8*, i8*)

; CHECK: declare i8* @strtok_r(i8*, i8* nocapture readonly, i8**) [[G0]]
declare i8* @strtok_r(i8*, i8*, i8**)

; CHECK: declare i64 @strtol(i8* readonly, i8** nocapture, i32) [[G0]]
declare i64 @strtol(i8*, i8**, i32)

; CHECK: declare x86_fp80 @strtold(i8* readonly, i8** nocapture) [[G0]]
declare x86_fp80 @strtold(i8*, i8**)

; CHECK: declare i64 @strtoll(i8* readonly, i8** nocapture, i32) [[G0]]
declare i64 @strtoll(i8*, i8**, i32)

; CHECK: declare i64 @strtoul(i8* readonly, i8** nocapture, i32) [[G0]]
declare i64 @strtoul(i8*, i8**, i32)

; CHECK: declare i64 @strtoull(i8* readonly, i8** nocapture, i32) [[G0]]
declare i64 @strtoull(i8*, i8**, i32)

; CHECK: declare i64 @strxfrm(i8* nocapture, i8* nocapture readonly, i64) [[G0]]
declare i64 @strxfrm(i8*, i8*, i64)

; CHECK: declare i32 @system(i8* nocapture readonly)
declare i32 @system(i8*)

; CHECK: declare double @tan(double)
declare double @tan(double)

; CHECK: declare float @tanf(float)
declare float @tanf(float)

; CHECK: declare double @tanh(double)
declare double @tanh(double)

; CHECK: declare float @tanhf(float)
declare float @tanhf(float)

; CHECK: declare x86_fp80 @tanhl(x86_fp80)
declare x86_fp80 @tanhl(x86_fp80)

; CHECK: declare x86_fp80 @tanl(x86_fp80)
declare x86_fp80 @tanl(x86_fp80)

; CHECK: declare i64 @times(%opaque* nocapture) [[G0]]
declare i64 @times(%opaque*)

; CHECK: declare noalias %opaque* @tmpfile() [[G0]]
declare %opaque* @tmpfile()

; CHECK-LINUX: declare noalias %opaque* @tmpfile64() [[G0]]
declare %opaque* @tmpfile64()

; CHECK: declare i32 @toascii(i32)
declare i32 @toascii(i32)

; CHECK: declare double @trunc(double)
declare double @trunc(double)

; CHECK: declare float @truncf(float)
declare float @truncf(float)

; CHECK: declare x86_fp80 @truncl(x86_fp80)
declare x86_fp80 @truncl(x86_fp80)

; CHECK: declare i32 @uname(%opaque* nocapture) [[G0]]
declare i32 @uname(%opaque*)

; CHECK: declare i32 @ungetc(i32, %opaque* nocapture) [[G0]]
declare i32 @ungetc(i32, %opaque*)

; CHECK: declare i32 @unlink(i8* nocapture readonly) [[G0]]
declare i32 @unlink(i8*)

; CHECK: declare i32 @unsetenv(i8* nocapture readonly) [[G0]]
declare i32 @unsetenv(i8*)

; CHECK: declare i32 @utime(i8* nocapture readonly, %opaque* nocapture readonly) [[G0]]
declare i32 @utime(i8*, %opaque*)

; CHECK: declare i32 @utimes(i8* nocapture readonly, %opaque* nocapture readonly) [[G0]]
declare i32 @utimes(i8*, %opaque*)

; CHECK: declare noalias i8* @valloc(i64) [[G0]]
declare i8* @valloc(i64)

; CHECK: declare i32 @vfprintf(%opaque* nocapture, i8* nocapture readonly, %opaque*) [[G0]]
declare i32 @vfprintf(%opaque*, i8*, %opaque*)

; CHECK: declare i32 @vfscanf(%opaque* nocapture, i8* nocapture readonly, %opaque*) [[G0]]
declare i32 @vfscanf(%opaque*, i8*, %opaque*)

; CHECK: declare i32 @vprintf(i8* nocapture readonly, %opaque*) [[G0]]
declare i32 @vprintf(i8*, %opaque*)

; CHECK: declare i32 @vscanf(i8* nocapture readonly, %opaque*) [[G0]]
declare i32 @vscanf(i8*, %opaque*)

; CHECK: declare i32 @vsnprintf(i8* nocapture, i64, i8* nocapture readonly, %opaque*) [[G0]]
declare i32 @vsnprintf(i8*, i64, i8*, %opaque*)

; CHECK: declare i32 @vsprintf(i8* nocapture, i8* nocapture readonly, %opaque*) [[G0]]
declare i32 @vsprintf(i8*, i8*, %opaque*)

; CHECK: declare i32 @vsscanf(i8* nocapture readonly, i8* nocapture readonly, %opaque*) [[G0]]
declare i32 @vsscanf(i8*, i8*, %opaque*)

; CHECK: declare i64 @write(i32, i8* nocapture readonly, i64)
declare i64 @write(i32, i8*, i64)


; memset_pattern16 isn't available everywhere.
; CHECK-DARWIN: declare void @memset_pattern16(i8* nocapture, i8* nocapture readonly, i64) [[G3:#[0-9]+]]
declare void @memset_pattern16(i8*, i8*, i64)


; CHECK: attributes [[G0]] = { nounwind }
; CHECK: attributes [[G1]] = { nounwind readonly }
; CHECK: attributes [[G2]] = { argmemonly nounwind readonly }
; CHECK-DARWIN: attributes [[G3]] = { argmemonly }
