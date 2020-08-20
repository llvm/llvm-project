; RUN: opt < %s -mtriple=x86_64-- -inferattrs -S | FileCheck -check-prefix=CHECK-UNKNOWN %s
; RUN: opt < %s -mtriple=x86_64-- -passes=inferattrs -S | FileCheck -check-prefix=CHECK-UNKNOWN %s
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.8.0 -inferattrs -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-DARWIN %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -inferattrs -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-LINUX %s
; RUN: opt < %s -mtriple=nvptx -inferattrs -S | FileCheck -check-prefix=CHECK-NVPTX %s

; operator new routines
declare i8* @_Znwj(i64 )
; CHECK: declare noalias nonnull i8* @_Znwj(i64) [[G0:#[0-9]+]]
declare i8* @_Znwm(i64)
; CHECK: declare noalias nonnull i8* @_Znwm(i64) [[G0]]

declare i32 @__nvvm_reflect(i8*)
; CHECK-NVPTX: declare i32 @__nvvm_reflect(i8*) [[G0:#[0-9]+]]
; CHECK-NVPTX: attributes [[G0]] = { nofree nounwind readnone }


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

; CHECK: declare i32 @abs(i32) [[G0]]
declare i32 @abs(i32)

; CHECK: declare i32 @access(i8* nocapture readonly, i32) [[G1:#[0-9]+]]
declare i32 @access(i8*, i32)

; CHECK: declare double @acos(double) [[G0]]
declare double @acos(double)

; CHECK: declare float @acosf(float) [[G0]]
declare float @acosf(float)

; CHECK: declare double @acosh(double) [[G0]]
declare double @acosh(double)

; CHECK: declare float @acoshf(float) [[G0]]
declare float @acoshf(float)

; CHECK: declare x86_fp80 @acoshl(x86_fp80) [[G0]]
declare x86_fp80 @acoshl(x86_fp80)

; CHECK: declare x86_fp80 @acosl(x86_fp80) [[G0]]
declare x86_fp80 @acosl(x86_fp80)

; CHECK: declare double @asin(double) [[G0]]
declare double @asin(double)

; CHECK: declare float @asinf(float) [[G0]]
declare float @asinf(float)

; CHECK: declare double @asinh(double) [[G0]]
declare double @asinh(double)

; CHECK: declare float @asinhf(float) [[G0]]
declare float @asinhf(float)

; CHECK: declare x86_fp80 @asinhl(x86_fp80) [[G0]]
declare x86_fp80 @asinhl(x86_fp80)

; CHECK: declare x86_fp80 @asinl(x86_fp80) [[G0]]
declare x86_fp80 @asinl(x86_fp80)

; CHECK: declare double @atan(double) [[G0]]
declare double @atan(double)

; CHECK: declare double @atan2(double, double) [[G0]]
declare double @atan2(double, double)

; CHECK: declare float @atan2f(float, float) [[G0]]
declare float @atan2f(float, float)

; CHECK: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[G0]]
declare x86_fp80 @atan2l(x86_fp80, x86_fp80)

; CHECK: declare float @atanf(float) [[G0]]
declare float @atanf(float)

; CHECK: declare double @atanh(double) [[G0]]
declare double @atanh(double)

; CHECK: declare float @atanhf(float) [[G0]]
declare float @atanhf(float)

; CHECK: declare x86_fp80 @atanhl(x86_fp80) [[G0]]
declare x86_fp80 @atanhl(x86_fp80)

; CHECK: declare x86_fp80 @atanl(x86_fp80) [[G0]]
declare x86_fp80 @atanl(x86_fp80)

; CHECK: declare double @atof(i8* nocapture) [[G2:#[0-9]+]]
declare double @atof(i8*)

; CHECK: declare i32 @atoi(i8* nocapture) [[G2]]
declare i32 @atoi(i8*)

; CHECK: declare i64 @atol(i8* nocapture) [[G2]]
declare i64 @atol(i8*)

; CHECK: declare i64 @atoll(i8* nocapture) [[G2]]
declare i64 @atoll(i8*)

; CHECK-LINUX: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G2]]
; CHECK-DARWIN-NOT: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G2]]
; CHECK-UNKNOWN-NOT: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G2]]
; CHECK-NVPTX-NOT: declare i32 @bcmp(i8* nocapture, i8* nocapture, i64) [[G2]]
declare i32 @bcmp(i8*, i8*, i64)

; CHECK: declare void @bcopy(i8* nocapture readonly, i8* nocapture, i64) [[G1]]
declare void @bcopy(i8*, i8*, i64)

; CHECK: declare void @bzero(i8* nocapture, i64) [[G1]]
declare void @bzero(i8*, i64)

; CHECK: declare noalias i8* @calloc(i64, i64) [[G1]]
declare i8* @calloc(i64, i64)

; CHECK: declare double @cbrt(double) [[G0]]
declare double @cbrt(double)

; CHECK: declare float @cbrtf(float) [[G0]]
declare float @cbrtf(float)

; CHECK: declare x86_fp80 @cbrtl(x86_fp80) [[G0]]
declare x86_fp80 @cbrtl(x86_fp80)

; CHECK: declare double @ceil(double) [[G0]]
declare double @ceil(double)

; CHECK: declare float @ceilf(float) [[G0]]
declare float @ceilf(float)

; CHECK: declare x86_fp80 @ceill(x86_fp80) [[G0]]
declare x86_fp80 @ceill(x86_fp80)

; CHECK: declare i32 @chmod(i8* nocapture readonly, i16 zeroext) [[G1]]
declare i32 @chmod(i8*, i16 zeroext)

; CHECK: declare i32 @chown(i8* nocapture readonly, i32, i32) [[G1]]
declare i32 @chown(i8*, i32, i32)

; CHECK: declare void @clearerr(%opaque* nocapture) [[G1]]
declare void @clearerr(%opaque*)

; CHECK: declare i32 @closedir(%opaque* nocapture) [[G1]]
declare i32 @closedir(%opaque*)

; CHECK: declare double @copysign(double, double) [[G0]]
declare double @copysign(double, double)

; CHECK: declare float @copysignf(float, float) [[G0]]
declare float @copysignf(float, float)

; CHECK: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[G0]]
declare x86_fp80 @copysignl(x86_fp80, x86_fp80)

; CHECK: declare double @cos(double) [[G0]]
declare double @cos(double)

; CHECK: declare float @cosf(float) [[G0]]
declare float @cosf(float)

; CHECK: declare double @cosh(double) [[G0]]
declare double @cosh(double)

; CHECK: declare float @coshf(float) [[G0]]
declare float @coshf(float)

; CHECK: declare x86_fp80 @coshl(x86_fp80) [[G0]]
declare x86_fp80 @coshl(x86_fp80)

; CHECK: declare x86_fp80 @cosl(x86_fp80) [[G0]]
declare x86_fp80 @cosl(x86_fp80)

; CHECK: declare i8* @ctermid(i8* nocapture) [[G1]]
declare i8* @ctermid(i8*)

; CHECK: declare double @exp(double) [[G0]]
declare double @exp(double)

; CHECK: declare double @exp2(double) [[G0]]
declare double @exp2(double)

; CHECK: declare float @exp2f(float) [[G0]]
declare float @exp2f(float)

; CHECK: declare x86_fp80 @exp2l(x86_fp80) [[G0]]
declare x86_fp80 @exp2l(x86_fp80)

; CHECK: declare float @expf(float) [[G0]]
declare float @expf(float)

; CHECK: declare x86_fp80 @expl(x86_fp80) [[G0]]
declare x86_fp80 @expl(x86_fp80)

; CHECK: declare double @expm1(double) [[G0]]
declare double @expm1(double)

; CHECK: declare float @expm1f(float) [[G0]]
declare float @expm1f(float)

; CHECK: declare x86_fp80 @expm1l(x86_fp80) [[G0]]
declare x86_fp80 @expm1l(x86_fp80)

; CHECK: declare double @fabs(double) [[G0]]
declare double @fabs(double)

; CHECK: declare float @fabsf(float) [[G0]]
declare float @fabsf(float)

; CHECK: declare x86_fp80 @fabsl(x86_fp80) [[G0]]
declare x86_fp80 @fabsl(x86_fp80)

; CHECK: declare noundef i32 @fclose(%opaque* nocapture noundef) [[G1]]
declare i32 @fclose(%opaque*)

; CHECK: declare noalias noundef %opaque* @fdopen(i32 noundef, i8* nocapture noundef readonly) [[G1]]
declare %opaque* @fdopen(i32, i8*)

; CHECK: declare noundef i32 @feof(%opaque* nocapture noundef) [[G1]]
declare i32 @feof(%opaque*)

; CHECK: declare noundef i32 @ferror(%opaque* nocapture noundef) [[G2]]
declare i32 @ferror(%opaque*)

; CHECK: declare noundef i32 @fflush(%opaque* nocapture noundef) [[G1]]
declare i32 @fflush(%opaque*)

; CHECK: declare i32 @ffs(i32) [[G0]]
declare i32 @ffs(i32)

; CHECK: declare i32 @ffsl(i64) [[G0]]
declare i32 @ffsl(i64)

; CHECK: declare i32 @ffsll(i64) [[G0]]
declare i32 @ffsll(i64)

; CHECK: declare noundef i32 @fgetc(%opaque* nocapture noundef) [[G1]]
declare i32 @fgetc(%opaque*)

; CHECK: declare noundef i32 @fgetpos(%opaque* nocapture noundef, i64* nocapture noundef) [[G1]]
declare i32 @fgetpos(%opaque*, i64*)

; CHECK: declare noundef i8* @fgets(i8* noundef, i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i8* @fgets(i8*, i32, %opaque*)

; CHECK: declare noundef i32 @fileno(%opaque* nocapture noundef) [[G1]]
declare i32 @fileno(%opaque*)

; CHECK: declare void @flockfile(%opaque* nocapture noundef) [[G1]]
declare void @flockfile(%opaque*)

; CHECK: declare double @floor(double) [[G0]]
declare double @floor(double)

; CHECK: declare float @floorf(float) [[G0]]
declare float @floorf(float)

; CHECK: declare x86_fp80 @floorl(x86_fp80) [[G0]]
declare x86_fp80 @floorl(x86_fp80)

; CHECK: declare i32 @fls(i32)
declare i32 @fls(i32)

; CHECK: declare i32 @flsl(i64)
declare i32 @flsl(i64)

; CHECK: declare i32 @flsll(i64)
declare i32 @flsll(i64)

; CHECK: declare double @fmax(double, double) [[G0]]
declare double @fmax(double, double)

; CHECK: declare float @fmaxf(float, float) [[G0]]
declare float @fmaxf(float, float)

; CHECK: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[G0]]
declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)

; CHECK: declare double @fmin(double, double) [[G0]]
declare double @fmin(double, double)

; CHECK: declare float @fminf(float, float) [[G0]]
declare float @fminf(float, float)

; CHECK: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[G0]]
declare x86_fp80 @fminl(x86_fp80, x86_fp80)

; CHECK: declare double @fmod(double, double) [[G0]]
declare double @fmod(double, double)

; CHECK: declare float @fmodf(float, float) [[G0]]
declare float @fmodf(float, float)

; CHECK: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[G0]]
declare x86_fp80 @fmodl(x86_fp80, x86_fp80)

; CHECK: declare noalias noundef %opaque* @fopen(i8* nocapture noundef readonly, i8* nocapture noundef readonly) [[G1]]
declare %opaque* @fopen(i8*, i8*)

; CHECK: declare noundef i32 @fprintf(%opaque* nocapture noundef, i8* nocapture noundef readonly, ...) [[G1]]
declare i32 @fprintf(%opaque*, i8*, ...)

; CHECK: declare noundef i32 @fputc(i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i32 @fputc(i32, %opaque*)

; CHECK: declare noundef i32 @fputs(i8* nocapture noundef readonly, %opaque* nocapture noundef) [[G1]]
declare i32 @fputs(i8*, %opaque*)

; CHECK: declare noundef i64 @fread(i8* nocapture noundef, i64 noundef, i64 noundef, %opaque* nocapture noundef) [[G1]]
declare i64 @fread(i8*, i64, i64, %opaque*)

; CHECK: declare void @free(i8* nocapture) [[G3:#[0-9]+]]
declare void @free(i8*)

; CHECK: declare double @frexp(double, i32* nocapture) [[G1]]
declare double @frexp(double, i32*)

; CHECK: declare float @frexpf(float, i32* nocapture) [[G1]]
declare float @frexpf(float, i32*)

; CHECK: declare x86_fp80 @frexpl(x86_fp80, i32* nocapture) [[G1]]
declare x86_fp80 @frexpl(x86_fp80, i32*)

; CHECK: declare noundef i32 @fscanf(%opaque* nocapture noundef, i8* nocapture noundef readonly, ...) [[G1]]
declare i32 @fscanf(%opaque*, i8*, ...)

; CHECK: declare noundef i32 @fseek(%opaque* nocapture noundef, i64 noundef, i32 noundef) [[G1]]
declare i32 @fseek(%opaque*, i64, i32)

; CHECK: declare noundef i32 @fseeko(%opaque* nocapture noundef, i64 noundef, i32 noundef) [[G1]]
declare i32 @fseeko(%opaque*, i64, i32)

; CHECK-LINUX: declare noundef i32 @fseeko64(%opaque* nocapture noundef, i64 noundef, i32 noundef) [[G1]]
declare i32 @fseeko64(%opaque*, i64, i32)

; CHECK: declare noundef i32 @fsetpos(%opaque* nocapture noundef, i64* noundef) [[G1]]
declare i32 @fsetpos(%opaque*, i64*)

; CHECK: declare noundef i32 @fstat(i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i32 @fstat(i32, %opaque*)

; CHECK-LINUX: declare noundef i32 @fstat64(i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i32 @fstat64(i32, %opaque*)

; CHECK: declare noundef i32 @fstatvfs(i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i32 @fstatvfs(i32, %opaque*)

; CHECK-LINUX: declare noundef i32 @fstatvfs64(i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i32 @fstatvfs64(i32, %opaque*)

; CHECK: declare noundef i64 @ftell(%opaque* nocapture noundef) [[G1]]
declare i64 @ftell(%opaque*)

; CHECK: declare noundef i64 @ftello(%opaque* nocapture noundef) [[G1]]
declare i64 @ftello(%opaque*)

; CHECK-LINUX: declare noundef i64 @ftello64(%opaque* nocapture noundef) [[G1]]
declare i64 @ftello64(%opaque*)

; CHECK: declare noundef i32 @ftrylockfile(%opaque* nocapture noundef) [[G1]]
declare i32 @ftrylockfile(%opaque*)

; CHECK: declare void @funlockfile(%opaque* nocapture noundef) [[G1]]
declare void @funlockfile(%opaque*)

; CHECK: declare noundef i64 @fwrite(i8* nocapture noundef, i64 noundef, i64 noundef, %opaque* nocapture noundef) [[G1]]
declare i64 @fwrite(i8*, i64, i64, %opaque*)

; CHECK: declare noundef i32 @getc(%opaque* nocapture noundef) [[G1]]
declare i32 @getc(%opaque*)

; CHECK: declare noundef i32 @getc_unlocked(%opaque* nocapture noundef) [[G1]]
declare i32 @getc_unlocked(%opaque*)

; CHECK: declare noundef i32 @getchar() [[G1]]
declare i32 @getchar()

; CHECK: declare noundef i32 @getchar_unlocked() [[G1]]
declare i32 @getchar_unlocked()

; CHECK: declare i8* @getenv(i8* nocapture) [[G2]]
declare i8* @getenv(i8*)

; CHECK: declare i32 @getitimer(i32, %opaque* nocapture) [[G1]]
declare i32 @getitimer(i32, %opaque*)

; CHECK: declare i32 @getlogin_r(i8* nocapture, i64) [[G1]]
declare i32 @getlogin_r(i8*, i64)

; CHECK: declare %opaque* @getpwnam(i8* nocapture readonly) [[G1]]
declare %opaque* @getpwnam(i8*)

; CHECK: declare noundef i8* @gets(i8* noundef) [[G1]]
declare i8* @gets(i8*)

; CHECK: declare i32 @gettimeofday(%opaque* nocapture, i8* nocapture) [[G1]]
declare i32 @gettimeofday(%opaque*, i8*)

; CHECK: declare i32 @isascii(i32) [[G0]]
declare i32 @isascii(i32)

; CHECK: declare i32 @isdigit(i32) [[G0]]
declare i32 @isdigit(i32)

; CHECK: declare i64 @labs(i64) [[G0]]
declare i64 @labs(i64)

; CHECK: declare i32 @lchown(i8* nocapture readonly, i32, i32) [[G1]]
declare i32 @lchown(i8*, i32, i32)

; CHECK: declare double @ldexp(double, i32) [[G0]]
declare double @ldexp(double, i32)

; CHECK: declare float @ldexpf(float, i32) [[G0]]
declare float @ldexpf(float, i32)

; CHECK: declare x86_fp80 @ldexpl(x86_fp80, i32) [[G0]]
declare x86_fp80 @ldexpl(x86_fp80, i32)

; CHECK: declare i64 @llabs(i64) [[G0]]
declare i64 @llabs(i64)

; CHECK: declare double @log(double) [[G0]]
declare double @log(double)

; CHECK: declare double @log10(double) [[G0]]
declare double @log10(double)

; CHECK: declare float @log10f(float) [[G0]]
declare float @log10f(float)

; CHECK: declare x86_fp80 @log10l(x86_fp80) [[G0]]
declare x86_fp80 @log10l(x86_fp80)

; CHECK: declare double @log1p(double) [[G0]]
declare double @log1p(double)

; CHECK: declare float @log1pf(float) [[G0]]
declare float @log1pf(float)

; CHECK: declare x86_fp80 @log1pl(x86_fp80) [[G0]]
declare x86_fp80 @log1pl(x86_fp80)

; CHECK: declare double @log2(double) [[G0]]
declare double @log2(double)

; CHECK: declare float @log2f(float) [[G0]]
declare float @log2f(float)

; CHECK: declare x86_fp80 @log2l(x86_fp80) [[G0]]
declare x86_fp80 @log2l(x86_fp80)

; CHECK: declare double @logb(double) [[G0]]
declare double @logb(double)

; CHECK: declare float @logbf(float) [[G0]]
declare float @logbf(float)

; CHECK: declare x86_fp80 @logbl(x86_fp80) [[G0]]
declare x86_fp80 @logbl(x86_fp80)

; CHECK: declare float @logf(float) [[G0]]
declare float @logf(float)

; CHECK: declare x86_fp80 @logl(x86_fp80) [[G0]]
declare x86_fp80 @logl(x86_fp80)

; CHECK: declare i32 @lstat(i8* nocapture readonly, %opaque* nocapture) [[G1]]
declare i32 @lstat(i8*, %opaque*)

; CHECK-LINUX: declare i32 @lstat64(i8* nocapture readonly, %opaque* nocapture) [[G1]]
declare i32 @lstat64(i8*, %opaque*)

; CHECK: declare noalias i8* @malloc(i64) [[G1]]
declare i8* @malloc(i64)

; CHECK-LINUX: declare noalias i8* @memalign(i64, i64) [[G0]]
declare i8* @memalign(i64, i64)

; CHECK: declare i8* @memccpy(i8* noalias, i8* noalias nocapture readonly, i32, i64) [[G1]]
declare i8* @memccpy(i8*, i8*, i32, i64)

; CHECK: declare i8* @memchr(i8*, i32, i64) [[G2]]
declare i8* @memchr(i8*, i32, i64)

; CHECK: declare i32 @memcmp(i8* nocapture, i8* nocapture, i64) [[G2]]
declare i32 @memcmp(i8*, i8*, i64)

; CHECK: declare i8* @memcpy(i8* noalias returned, i8* noalias nocapture readonly, i64) [[G1]]
declare i8* @memcpy(i8*, i8*, i64)

; CHECK: declare i8* @mempcpy(i8* noalias, i8* noalias nocapture readonly, i64) [[G1]]
declare i8* @mempcpy(i8*, i8*, i64)

; CHECK: declare i8* @memmove(i8* returned, i8* nocapture readonly, i64) [[G1]]
declare i8* @memmove(i8*, i8*, i64)

; CHECK: declare i8* @memset(i8*, i32, i64) [[G0]]
declare i8* @memset(i8*, i32, i64)

; CHECK: declare i32 @mkdir(i8* nocapture readonly, i16 zeroext) [[G1]]
declare i32 @mkdir(i8*, i16 zeroext)

; CHECK: declare i64 @mktime(%opaque* nocapture) [[G1]]
declare i64 @mktime(%opaque*)

; CHECK: declare double @modf(double, double* nocapture) [[G1]]
declare double @modf(double, double*)

; CHECK: declare float @modff(float, float* nocapture) [[G1]]
declare float @modff(float, float*)

; CHECK: declare x86_fp80 @modfl(x86_fp80, x86_fp80* nocapture) [[G1]]
declare x86_fp80 @modfl(x86_fp80, x86_fp80*)

; CHECK: declare double @nearbyint(double) [[G0]]
declare double @nearbyint(double)

; CHECK: declare float @nearbyintf(float) [[G0]]
declare float @nearbyintf(float)

; CHECK: declare x86_fp80 @nearbyintl(x86_fp80) [[G0]]
declare x86_fp80 @nearbyintl(x86_fp80)

; CHECK: declare noundef i32 @open(i8* nocapture noundef readonly, i32 noundef, ...) [[G0]]
declare i32 @open(i8*, i32, ...)

; CHECK-LINUX: declare noundef i32 @open64(i8* nocapture noundef readonly, i32 noundef, ...) [[G0]]
declare i32 @open64(i8*, i32, ...)

; CHECK: declare noalias %opaque* @opendir(i8* nocapture readonly) [[G1]]
declare %opaque* @opendir(i8*)

; CHECK: declare i32 @pclose(%opaque* nocapture) [[G1]]
declare i32 @pclose(%opaque*)

; CHECK: declare void @perror(i8* nocapture noundef readonly) [[G1]]
declare void @perror(i8*)

; CHECK: declare noalias %opaque* @popen(i8* nocapture readonly, i8* nocapture readonly) [[G1]]
declare %opaque* @popen(i8*, i8*)

; CHECK: declare i32 @posix_memalign(i8**, i64, i64) [[G0]]
declare i32 @posix_memalign(i8**, i64, i64)

; CHECK: declare double @pow(double, double) [[G0]]
declare double @pow(double, double)

; CHECK: declare float @powf(float, float) [[G0]]
declare float @powf(float, float)

; CHECK: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[G0]]
declare x86_fp80 @powl(x86_fp80, x86_fp80)

; CHECK: declare noundef i64 @pread(i32 noundef, i8* nocapture noundef, i64 noundef, i64 noundef) [[G0]]
declare i64 @pread(i32, i8*, i64, i64)

; CHECK: declare noundef i32 @printf(i8* nocapture noundef readonly, ...) [[G1]]
declare i32 @printf(i8*, ...)

; CHECK: declare noundef i32 @putc(i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i32 @putc(i32, %opaque*)

; CHECK: declare noundef i32 @putchar(i32 noundef) [[G1]]
declare i32 @putchar(i32)

; CHECK: declare noundef i32 @putchar_unlocked(i32 noundef) [[G1]]
declare i32 @putchar_unlocked(i32)

; CHECK: declare noundef i32 @puts(i8* nocapture noundef readonly) [[G1]]
declare i32 @puts(i8*)

; CHECK: declare noundef i64 @pwrite(i32 noundef, i8* nocapture noundef readonly, i64 noundef, i64 noundef) [[G0]]
declare i64 @pwrite(i32, i8*, i64, i64)

; CHECK: declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)* nocapture) [[G0]]
declare void @qsort(i8*, i64, i64, i32 (i8*, i8*)*)

; CHECK: declare noundef i64 @read(i32 noundef, i8* nocapture noundef, i64 noundef) [[G0]]
declare i64 @read(i32, i8*, i64)

; CHECK: declare i64 @readlink(i8* nocapture readonly, i8* nocapture, i64) [[G1]]
declare i64 @readlink(i8*, i8*, i64)

; CHECK: declare noalias i8* @realloc(i8* nocapture, i64) [[G3]]
declare i8* @realloc(i8*, i64)

; CHECK: declare i8* @reallocf(i8*, i64)
declare i8* @reallocf(i8*, i64)

; CHECK: declare i8* @realpath(i8* nocapture readonly, i8*) [[G1]]
declare i8* @realpath(i8*, i8*)

; CHECK: declare i32 @remove(i8* nocapture readonly) [[G1]]
declare i32 @remove(i8*)

; CHECK: declare i32 @rename(i8* nocapture readonly, i8* nocapture readonly) [[G1]]
declare i32 @rename(i8*, i8*)

; CHECK: declare void @rewind(%opaque* nocapture noundef) [[G1]]
declare void @rewind(%opaque*)

; CHECK: declare double @rint(double) [[G0]]
declare double @rint(double)

; CHECK: declare float @rintf(float) [[G0]]
declare float @rintf(float)

; CHECK: declare x86_fp80 @rintl(x86_fp80) [[G0]]
declare x86_fp80 @rintl(x86_fp80)

; CHECK: declare i32 @rmdir(i8* nocapture readonly) [[G1]]
declare i32 @rmdir(i8*)

; CHECK: declare double @round(double) [[G0]]
declare double @round(double)

; CHECK: declare float @roundf(float) [[G0]]
declare float @roundf(float)

; CHECK: declare x86_fp80 @roundl(x86_fp80) [[G0]]
declare x86_fp80 @roundl(x86_fp80)

; CHECK: declare noundef i32 @scanf(i8* nocapture noundef readonly, ...) [[G1]]
declare i32 @scanf(i8*, ...)

; CHECK: declare void @setbuf(%opaque* nocapture, i8*) [[G1]]
declare void @setbuf(%opaque*, i8*)

; CHECK: declare i32 @setitimer(i32, %opaque* nocapture readonly, %opaque* nocapture) [[G1]]
declare i32 @setitimer(i32, %opaque*, %opaque*)

; CHECK: declare i32 @setvbuf(%opaque* nocapture, i8*, i32, i64) [[G1]]
declare i32 @setvbuf(%opaque*, i8*, i32, i64)

; CHECK: declare double @sin(double) [[G0]]
declare double @sin(double)

; CHECK: declare float @sinf(float) [[G0]]
declare float @sinf(float)

; CHECK: declare double @sinh(double) [[G0]]
declare double @sinh(double)

; CHECK: declare float @sinhf(float) [[G0]]
declare float @sinhf(float)

; CHECK: declare x86_fp80 @sinhl(x86_fp80) [[G0]]
declare x86_fp80 @sinhl(x86_fp80)

; CHECK: declare x86_fp80 @sinl(x86_fp80) [[G0]]
declare x86_fp80 @sinl(x86_fp80)

; CHECK: declare noundef i32 @snprintf(i8* noalias nocapture noundef, i64 noundef, i8* nocapture noundef readonly, ...) [[G1]]
declare i32 @snprintf(i8*, i64, i8*, ...)

; CHECK: declare noundef i32 @sprintf(i8* noalias nocapture noundef, i8* nocapture noundef readonly, ...) [[G1]]
declare i32 @sprintf(i8*, i8*, ...)

; CHECK: declare double @sqrt(double) [[G0]]
declare double @sqrt(double)

; CHECK: declare float @sqrtf(float) [[G0]]
declare float @sqrtf(float)

; CHECK: declare x86_fp80 @sqrtl(x86_fp80) [[G0]]
declare x86_fp80 @sqrtl(x86_fp80)

; CHECK: declare noundef i32 @sscanf(i8* nocapture noundef readonly, i8* nocapture noundef readonly, ...) [[G1]]
declare i32 @sscanf(i8*, i8*, ...)

; CHECK: declare i32 @stat(i8* nocapture readonly, %opaque* nocapture) [[G1]]
declare i32 @stat(i8*, %opaque*)

; CHECK-LINUX: declare i32 @stat64(i8* nocapture readonly, %opaque* nocapture) [[G1]]
declare i32 @stat64(i8*, %opaque*)

; CHECK: declare i32 @statvfs(i8* nocapture readonly, %opaque* nocapture) [[G1]]
declare i32 @statvfs(i8*, %opaque*)

; CHECK-LINUX: declare i32 @statvfs64(i8* nocapture readonly, %opaque* nocapture) [[G1]]
declare i32 @statvfs64(i8*, %opaque*)

; CHECK: declare i8* @stpcpy(i8*, i8* nocapture readonly) [[G1]]
declare i8* @stpcpy(i8*, i8*)

; CHECK: declare i8* @stpncpy(i8*, i8* nocapture readonly, i64) [[G1]]
declare i8* @stpncpy(i8*, i8*, i64)

; CHECK: declare i32 @strcasecmp(i8* nocapture, i8* nocapture) [[G2]]
declare i32 @strcasecmp(i8*, i8*)

; CHECK: declare i8* @strcat(i8* returned, i8* nocapture readonly) [[G1]]
declare i8* @strcat(i8*, i8*)

; CHECK: declare i8* @strchr(i8*, i32) [[G2]]
declare i8* @strchr(i8*, i32)

; CHECK: declare i32 @strcmp(i8* nocapture, i8* nocapture) [[G2]]
declare i32 @strcmp(i8*, i8*)

; CHECK: declare i32 @strcoll(i8* nocapture, i8* nocapture) [[G2]]
declare i32 @strcoll(i8*, i8*)

; CHECK: declare i8* @strcpy(i8* noalias returned, i8* noalias nocapture readonly) [[G1]]
declare i8* @strcpy(i8*, i8*)

; CHECK: declare i64 @strcspn(i8* nocapture, i8* nocapture) [[G2]]
declare i64 @strcspn(i8*, i8*)

; CHECK: declare noalias i8* @strdup(i8* nocapture readonly) [[G1]]
declare i8* @strdup(i8*)

; CHECK: declare i64 @strlen(i8* nocapture) [[G4:#[0-9]+]]
declare i64 @strlen(i8*)

; CHECK: declare i32 @strncasecmp(i8* nocapture, i8* nocapture, i64) [[G2]]
declare i32 @strncasecmp(i8*, i8*, i64)

; CHECK: declare i8* @strncat(i8* returned, i8* nocapture readonly, i64) [[G1]]
declare i8* @strncat(i8*, i8*, i64)

; CHECK: declare i32 @strncmp(i8* nocapture, i8* nocapture, i64) [[G2]]
declare i32 @strncmp(i8*, i8*, i64)

; CHECK: declare i8* @strncpy(i8* noalias returned, i8* noalias nocapture readonly, i64) [[G1]]
declare i8* @strncpy(i8*, i8*, i64)

; CHECK: declare noalias i8* @strndup(i8* nocapture readonly, i64) [[G1]]
declare i8* @strndup(i8*, i64)

; CHECK: declare i64 @strnlen(i8*, i64) [[G0]]
declare i64 @strnlen(i8*, i64)

; CHECK: declare i8* @strpbrk(i8*, i8* nocapture) [[G2]]
declare i8* @strpbrk(i8*, i8*)

; CHECK: declare i8* @strrchr(i8*, i32) [[G2]]
declare i8* @strrchr(i8*, i32)

; CHECK: declare i64 @strspn(i8* nocapture, i8* nocapture) [[G2]]
declare i64 @strspn(i8*, i8*)

; CHECK: declare i8* @strstr(i8*, i8* nocapture) [[G2]]
declare i8* @strstr(i8*, i8*)

; CHECK: declare double @strtod(i8* readonly, i8** nocapture) [[G1]]
declare double @strtod(i8*, i8**)

; CHECK: declare float @strtof(i8* readonly, i8** nocapture) [[G1]]
declare float @strtof(i8*, i8**)

; CHECK: declare i8* @strtok(i8*, i8* nocapture readonly) [[G1]]
declare i8* @strtok(i8*, i8*)

; CHECK: declare i8* @strtok_r(i8*, i8* nocapture readonly, i8**) [[G1]]
declare i8* @strtok_r(i8*, i8*, i8**)

; CHECK: declare i64 @strtol(i8* readonly, i8** nocapture, i32) [[G1]]
declare i64 @strtol(i8*, i8**, i32)

; CHECK: declare x86_fp80 @strtold(i8* readonly, i8** nocapture) [[G1]]
declare x86_fp80 @strtold(i8*, i8**)

; CHECK: declare i64 @strtoll(i8* readonly, i8** nocapture, i32) [[G1]]
declare i64 @strtoll(i8*, i8**, i32)

; CHECK: declare i64 @strtoul(i8* readonly, i8** nocapture, i32) [[G1]]
declare i64 @strtoul(i8*, i8**, i32)

; CHECK: declare i64 @strtoull(i8* readonly, i8** nocapture, i32) [[G1]]
declare i64 @strtoull(i8*, i8**, i32)

; CHECK: declare i64 @strxfrm(i8* nocapture, i8* nocapture readonly, i64) [[G1]]
declare i64 @strxfrm(i8*, i8*, i64)

; CHECK: declare i32 @system(i8* nocapture readonly) [[G0]]
declare i32 @system(i8*)

; CHECK: declare double @tan(double) [[G0]]
declare double @tan(double)

; CHECK: declare float @tanf(float) [[G0]]
declare float @tanf(float)

; CHECK: declare double @tanh(double) [[G0]]
declare double @tanh(double)

; CHECK: declare float @tanhf(float) [[G0]]
declare float @tanhf(float)

; CHECK: declare x86_fp80 @tanhl(x86_fp80) [[G0]]
declare x86_fp80 @tanhl(x86_fp80)

; CHECK: declare x86_fp80 @tanl(x86_fp80) [[G0]]
declare x86_fp80 @tanl(x86_fp80)

; CHECK: declare i64 @times(%opaque* nocapture) [[G1]]
declare i64 @times(%opaque*)

; CHECK: declare noalias %opaque* @tmpfile() [[G1]]
declare %opaque* @tmpfile()

; CHECK-LINUX: declare noalias %opaque* @tmpfile64() [[G1]]
declare %opaque* @tmpfile64()

; CHECK: declare i32 @toascii(i32) [[G0]]
declare i32 @toascii(i32)

; CHECK: declare double @trunc(double) [[G0]]
declare double @trunc(double)

; CHECK: declare float @truncf(float) [[G0]]
declare float @truncf(float)

; CHECK: declare x86_fp80 @truncl(x86_fp80) [[G0]]
declare x86_fp80 @truncl(x86_fp80)

; CHECK: declare i32 @uname(%opaque* nocapture) [[G1]]
declare i32 @uname(%opaque*)

; CHECK: declare noundef i32 @ungetc(i32 noundef, %opaque* nocapture noundef) [[G1]]
declare i32 @ungetc(i32, %opaque*)

; CHECK: declare i32 @unlink(i8* nocapture readonly) [[G1]]
declare i32 @unlink(i8*)

; CHECK: declare i32 @unsetenv(i8* nocapture readonly) [[G1]]
declare i32 @unsetenv(i8*)

; CHECK: declare i32 @utime(i8* nocapture readonly, %opaque* nocapture readonly) [[G1]]
declare i32 @utime(i8*, %opaque*)

; CHECK: declare i32 @utimes(i8* nocapture readonly, %opaque* nocapture readonly) [[G1]]
declare i32 @utimes(i8*, %opaque*)

; CHECK: declare noalias i8* @valloc(i64) [[G1]]
declare i8* @valloc(i64)

; CHECK: declare noundef i32 @vfprintf(%opaque* nocapture noundef, i8* nocapture noundef readonly, %opaque* noundef) [[G1]]
declare i32 @vfprintf(%opaque*, i8*, %opaque*)

; CHECK: declare noundef i32 @vfscanf(%opaque* nocapture noundef, i8* nocapture noundef readonly, %opaque* noundef) [[G1]]
declare i32 @vfscanf(%opaque*, i8*, %opaque*)

; CHECK: declare noundef i32 @vprintf(i8* nocapture noundef readonly, %opaque* noundef) [[G1]]
declare i32 @vprintf(i8*, %opaque*)

; CHECK: declare noundef i32 @vscanf(i8* nocapture noundef readonly, %opaque* noundef) [[G1]]
declare i32 @vscanf(i8*, %opaque*)

; CHECK: declare noundef i32 @vsnprintf(i8* nocapture noundef, i64 noundef, i8* nocapture noundef readonly, %opaque* noundef) [[G1]]
declare i32 @vsnprintf(i8*, i64, i8*, %opaque*)

; CHECK: declare noundef i32 @vsprintf(i8* nocapture noundef, i8* nocapture noundef readonly, %opaque* noundef) [[G1]]
declare i32 @vsprintf(i8*, i8*, %opaque*)

; CHECK: declare noundef i32 @vsscanf(i8* nocapture noundef readonly, i8* nocapture noundef readonly, %opaque* noundef) [[G1]]
declare i32 @vsscanf(i8*, i8*, %opaque*)

; CHECK: declare noundef i64 @write(i32 noundef, i8* nocapture noundef readonly, i64 noundef) [[G0]]
declare i64 @write(i32, i8*, i64)


; memset_pattern16 isn't available everywhere.
; CHECK-DARWIN: declare void @memset_pattern16(i8* nocapture, i8* nocapture readonly, i64) [[G5:#[0-9]+]]
declare void @memset_pattern16(i8*, i8*, i64)

; CHECK: attributes [[G0]] = { nofree }
; CHECK: attributes [[G1]] = { nofree nounwind }
; CHECK: attributes [[G2]] = { nofree nounwind readonly }
; CHECK: attributes [[G3]] = { nounwind }
; CHECK: attributes [[G4]] = { argmemonly nofree nounwind readonly }
; CHECK-DARWIN: attributes [[G5]] = { argmemonly nofree }
