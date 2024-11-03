; RUN: opt < %s -mtriple=x86_64-- -passes=inferattrs -S | FileCheck --match-full-lines --check-prefixes=CHECK,CHECK-NOLINUX,CHECK-OPEN,CHECK-UNKNOWN %s
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.8.0 -passes=inferattrs -S | FileCheck --match-full-lines --check-prefixes=CHECK,CHECK-KNOWN,CHECK-NOLINUX,CHECK-OPEN,CHECK-DARWIN %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=inferattrs -S | FileCheck --match-full-lines --check-prefixes=CHECK,CHECK-KNOWN,CHECK-LINUX %s
; RUN: opt < %s -mtriple=nvptx -passes=inferattrs -S | FileCheck --match-full-lines --check-prefixes=CHECK-NOLINUX,CHECK-NVPTX %s
; RUN: opt < %s -mtriple=powerpc-ibm-aix-xcoff -passes=inferattrs -S | FileCheck --match-full-lines --check-prefixes=CHECK-AIX %s

declare i32 @__nvvm_reflect(ptr)
; CHECK-NVPTX: declare noundef i32 @__nvvm_reflect(ptr noundef) [[NOFREE_NOUNWIND_READNONE:#[0-9]+]]


; Check all the libc functions (thereby also exercising the prototype check).
; Note that it's OK to modify these as attributes might be missing. These checks
; reflect the currently inferred attributes.

; Use an opaque pointer type for all the (possibly opaque) structs.
%opaque = type opaque

; CHECK-LINUX: declare double @__acos_finite(double) [[NOFREE:#[0-9]+]]
; CHECK-NOLINUX: declare double @__acos_finite(double)
declare double @__acos_finite(double)

; CHECK-LINUX: declare float @__acosf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__acosf_finite(float)
declare float @__acosf_finite(float)

; CHECK-LINUX: declare double @__acosh_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__acosh_finite(double)
declare double @__acosh_finite(double)

; CHECK-LINUX: declare float @__acoshf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__acoshf_finite(float)
declare float @__acoshf_finite(float)

; CHECK-LINUX: declare x86_fp80 @__acoshl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__acoshl_finite(x86_fp80)
declare x86_fp80 @__acoshl_finite(x86_fp80)

; CHECK-LINUX: declare x86_fp80 @__acosl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__acosl_finite(x86_fp80)
declare x86_fp80 @__acosl_finite(x86_fp80)

; CHECK-LINUX: declare double @__asin_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__asin_finite(double)
declare double @__asin_finite(double)

; CHECK-LINUX: declare float @__asinf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__asinf_finite(float)
declare float @__asinf_finite(float)

; CHECK-LINUX: declare x86_fp80 @__asinl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__asinl_finite(x86_fp80)
declare x86_fp80 @__asinl_finite(x86_fp80)

; CHECK-LINUX: declare double @__atan2_finite(double, double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__atan2_finite(double, double)
declare double @__atan2_finite(double, double)

; CHECK-LINUX: declare float @__atan2f_finite(float, float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__atan2f_finite(float, float)
declare float @__atan2f_finite(float, float)

; CHECK-LINUX: declare x86_fp80 @__atan2l_finite(x86_fp80, x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__atan2l_finite(x86_fp80, x86_fp80)
declare x86_fp80 @__atan2l_finite(x86_fp80, x86_fp80)

; CHECK-LINUX: declare double @__atanh_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__atanh_finite(double)
declare double @__atanh_finite(double)

; CHECK-LINUX: declare float @__atanhf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__atanhf_finite(float)
declare float @__atanhf_finite(float)

; CHECK-LINUX: declare x86_fp80 @__atanhl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__atanhl_finite(x86_fp80)
declare x86_fp80 @__atanhl_finite(x86_fp80)

; CHECK-LINUX: declare double @__cosh_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__cosh_finite(double)
declare double @__cosh_finite(double)

; CHECK-LINUX: declare float @__coshf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__coshf_finite(float)
declare float @__coshf_finite(float)

; CHECK-LINUX: declare x86_fp80 @__coshl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__coshl_finite(x86_fp80)
declare x86_fp80 @__coshl_finite(x86_fp80)

; CHECK: declare double @__cospi(double)
declare double @__cospi(double)

; CHECK: declare float @__cospif(float)
declare float @__cospif(float)

; CHECK-LINUX: declare double @__exp10_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__exp10_finite(double)
declare double @__exp10_finite(double)

; CHECK-LINUX: declare float @__exp10f_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__exp10f_finite(float)
declare float @__exp10f_finite(float)

; CHECK-LINUX: declare x86_fp80 @__exp10l_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__exp10l_finite(x86_fp80)
declare x86_fp80 @__exp10l_finite(x86_fp80)

; CHECK-LINUX: declare double @__exp2_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__exp2_finite(double)
declare double @__exp2_finite(double)

; CHECK-LINUX: declare float @__exp2f_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__exp2f_finite(float)
declare float @__exp2f_finite(float)

; CHECK-LINUX: declare x86_fp80 @__exp2l_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__exp2l_finite(x86_fp80)
declare x86_fp80 @__exp2l_finite(x86_fp80)

; CHECK-LINUX: declare double @__exp_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__exp_finite(double)
declare double @__exp_finite(double)

; CHECK-LINUX: declare float @__expf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__expf_finite(float)
declare float @__expf_finite(float)

; CHECK-LINUX: declare x86_fp80 @__expl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__expl_finite(x86_fp80)
declare x86_fp80 @__expl_finite(x86_fp80)

; CHECK-LINUX: declare double @__log10_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__log10_finite(double)
declare double @__log10_finite(double)

; CHECK-LINUX: declare float @__log10f_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__log10f_finite(float)
declare float @__log10f_finite(float)

; CHECK-LINUX: declare x86_fp80 @__log10l_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__log10l_finite(x86_fp80)
declare x86_fp80 @__log10l_finite(x86_fp80)

; CHECK-LINUX: declare double @__log2_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__log2_finite(double)
declare double @__log2_finite(double)

; CHECK-LINUX: declare float @__log2f_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__log2f_finite(float)
declare float @__log2f_finite(float)

; CHECK-LINUX: declare x86_fp80 @__log2l_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__log2l_finite(x86_fp80)
declare x86_fp80 @__log2l_finite(x86_fp80)

; CHECK-LINUX: declare double @__log_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__log_finite(double)
declare double @__log_finite(double)

; CHECK-LINUX: declare float @__logf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__logf_finite(float)
declare float @__logf_finite(float)

; CHECK-LINUX: declare x86_fp80 @__logl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__logl_finite(x86_fp80)
declare x86_fp80 @__logl_finite(x86_fp80)

; CHECK-LINUX: declare double @__pow_finite(double, double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__pow_finite(double, double)
declare double @__pow_finite(double, double)

; CHECK-LINUX: declare float @__powf_finite(float, float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__powf_finite(float, float)
declare float @__powf_finite(float, float)

; CHECK-LINUX: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80)
declare x86_fp80 @__powl_finite(x86_fp80, x86_fp80)

; CHECK-LINUX: declare double @__sinh_finite(double) [[NOFREE]]
; CHECK-NOLINUX: declare double @__sinh_finite(double)
declare double @__sinh_finite(double)

; CHECK-LINUX: declare float @__sinhf_finite(float) [[NOFREE]]
; CHECK-NOLINUX: declare float @__sinhf_finite(float)
declare float @__sinhf_finite(float)

; CHECK-LINUX: declare x86_fp80 @__sinhl_finite(x86_fp80) [[NOFREE]]
; CHECK-NOLINUX: declare x86_fp80 @__sinhl_finite(x86_fp80)
declare x86_fp80 @__sinhl_finite(x86_fp80)

; CHECK: declare double @__sinpi(double)
declare double @__sinpi(double)

; CHECK: declare float @__sinpif(float)
declare float @__sinpif(float)

; CHECK: declare i32 @abs(i32) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY:#[0-9]+]]
declare i32 @abs(i32)

; CHECK: declare noundef i32 @access(ptr nocapture noundef readonly, i32 noundef) [[NOFREE_NOUNWIND:#[0-9]+]]
declare i32 @access(ptr, i32)

; CHECK: declare double @acos(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @acos(double)

; CHECK: declare float @acosf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @acosf(float)

; CHECK: declare double @acosh(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @acosh(double)

; CHECK: declare float @acoshf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @acoshf(float)

; CHECK: declare x86_fp80 @acoshl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @acoshl(x86_fp80)

; CHECK: declare x86_fp80 @acosl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @acosl(x86_fp80)

; CHECK: declare noalias noundef ptr @aligned_alloc(i64 allocalign noundef, i64 noundef) [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCKIND_ALLOCUNINIT_ALLOCSIZE1_FAMILY_MALLOC:#[0-9]+]]
declare ptr @aligned_alloc(i64, i64)

; CHECK: declare double @asin(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @asin(double)

; CHECK: declare float @asinf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @asinf(float)

; CHECK: declare double @asinh(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @asinh(double)

; CHECK: declare float @asinhf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @asinhf(float)

; CHECK: declare x86_fp80 @asinhl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @asinhl(x86_fp80)

; CHECK: declare x86_fp80 @asinl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @asinl(x86_fp80)

; CHECK: declare double @atan(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @atan(double)

; CHECK: declare double @atan2(double, double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @atan2(double, double)

; CHECK: declare float @atan2f(float, float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @atan2f(float, float)

; CHECK: declare x86_fp80 @atan2l(x86_fp80, x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @atan2l(x86_fp80, x86_fp80)

; CHECK: declare float @atanf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @atanf(float)

; CHECK: declare double @atanh(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @atanh(double)

; CHECK: declare float @atanhf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @atanhf(float)

; CHECK: declare x86_fp80 @atanhl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @atanhl(x86_fp80)

; CHECK: declare x86_fp80 @atanl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @atanl(x86_fp80)

; CHECK: declare double @atof(ptr nocapture) [[NOFREE_NOUNWIND_READONLY_WILLRETURN:#[0-9]+]]
declare double @atof(ptr)

; CHECK: declare i32 @atoi(ptr nocapture) [[NOFREE_NOUNWIND_READONLY_WILLRETURN]]
declare i32 @atoi(ptr)

; CHECK: declare i64 @atol(ptr nocapture) [[NOFREE_NOUNWIND_READONLY_WILLRETURN]]
declare i64 @atol(ptr)

; CHECK: declare i64 @atoll(ptr nocapture) [[NOFREE_NOUNWIND_READONLY_WILLRETURN]]
declare i64 @atoll(ptr)

; CHECK-LINUX: declare i32 @bcmp(ptr nocapture, ptr nocapture, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY:#[0-9]+]]
; CHECK-NOLINUX: declare i32 @bcmp(ptr, ptr, i64){{$}}
declare i32 @bcmp(ptr, ptr, i64)

; CHECK: declare void @bcopy(ptr nocapture readonly, ptr nocapture writeonly, i64)  [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN:#[0-9]+]]
declare void @bcopy(ptr, ptr, i64)

; CHECK: declare void @bzero(ptr nocapture writeonly, i64)  [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare void @bzero(ptr, i64)

; CHECK: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCKIND_ALLOCZEROED_ALLOCSIZE01_FAMILY_MALLOC:#[0-9]+]]
declare ptr @calloc(i64, i64)

; CHECK-AIX: declare noalias noundef ptr @vec_calloc(i64 noundef, i64 noundef) [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCSIZE01_FAMILY_VEC_MALLOC:#[0-9]+]]
declare ptr @vec_calloc(i64, i64)

; CHECK: declare double @cbrt(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @cbrt(double)

; CHECK: declare float @cbrtf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @cbrtf(float)

; CHECK: declare x86_fp80 @cbrtl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @cbrtl(x86_fp80)

; CHECK: declare double @ceil(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @ceil(double)

; CHECK: declare float @ceilf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @ceilf(float)

; CHECK: declare x86_fp80 @ceill(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @ceill(x86_fp80)

; The second argument of int chmod(FILE*, mode_t) is a 32-bit int on most
; targets but it's a 16-bit short on Apple Darwin.  Use i16 here to verify
; the function is still recognized.
; FIXME: this should be tightened up to verify that only the type with
; the right size for the target matches.
; CHECK: declare noundef i32 @chmod(ptr nocapture noundef readonly, i16 noundef zeroext) [[NOFREE_NOUNWIND]]
declare i32 @chmod(ptr, i16 zeroext)

; CHECK: declare noundef i32 @chown(ptr nocapture noundef readonly, i32 noundef, i32 noundef) [[NOFREE_NOUNWIND]]
declare i32 @chown(ptr, i32, i32)

; CHECK: declare void @clearerr(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare void @clearerr(ptr)

; CHECK: declare noundef i32 @closedir(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @closedir(ptr)

; CHECK: declare double @copysign(double, double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @copysign(double, double)

; CHECK: declare float @copysignf(float, float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @copysignf(float, float)

; CHECK: declare x86_fp80 @copysignl(x86_fp80, x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @copysignl(x86_fp80, x86_fp80)

; CHECK: declare double @cos(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @cos(double)

; CHECK: declare float @cosf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @cosf(float)

; CHECK: declare double @cosh(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @cosh(double)

; CHECK: declare float @coshf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @coshf(float)

; CHECK: declare x86_fp80 @coshl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @coshl(x86_fp80)

; CHECK: declare x86_fp80 @cosl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @cosl(x86_fp80)

; CHECK: declare noundef ptr @ctermid(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare ptr @ctermid(ptr)

; CHECK: declare double @exp(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @exp(double)

; CHECK: declare double @exp2(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @exp2(double)

; CHECK: declare float @exp2f(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @exp2f(float)

; CHECK: declare x86_fp80 @exp2l(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @exp2l(x86_fp80)

; CHECK: declare float @expf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @expf(float)

; CHECK: declare x86_fp80 @expl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @expl(x86_fp80)

; CHECK: declare double @expm1(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @expm1(double)

; CHECK: declare float @expm1f(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @expm1f(float)

; CHECK: declare x86_fp80 @expm1l(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @expm1l(x86_fp80)

; CHECK: declare double @fabs(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @fabs(double)

; CHECK: declare float @fabsf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @fabsf(float)

; CHECK: declare x86_fp80 @fabsl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @fabsl(x86_fp80)

; CHECK: declare noundef i32 @fclose(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fclose(ptr)

; CHECK: declare noalias noundef ptr @fdopen(i32 noundef, ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare ptr @fdopen(i32, ptr)

; CHECK: declare noundef i32 @feof(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @feof(ptr)

; CHECK: declare noundef i32 @ferror(ptr nocapture noundef) [[NOFREE_NOUNWIND_READONLY:#[0-9]+]]
declare i32 @ferror(ptr)

; CHECK: declare noundef i32 @fflush(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fflush(ptr)

; CHECK: declare i32 @ffs(i32) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare i32 @ffs(i32)

; CHECK-KNOWN: declare i32 @ffsl(i64) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
; CHECK-UNKNOWN: declare i32 @ffsl(i64){{$}}
declare i32 @ffsl(i64)

; CHECK-KNOWN: declare i32 @ffsll(i64) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
; CHECK-UNKNOWN: declare i32 @ffsll(i64){{$}}
declare i32 @ffsll(i64)

; CHECK: declare noundef i32 @fgetc(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fgetc(ptr)

; CHECK: declare noundef i32 @fgetpos(ptr nocapture noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fgetpos(ptr, ptr)

; CHECK: declare noundef ptr @fgets(ptr noundef, i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare ptr @fgets(ptr, i32, ptr)

; CHECK: declare noundef i32 @fileno(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fileno(ptr)

; CHECK: declare void @flockfile(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare void @flockfile(ptr)

; CHECK: declare double @floor(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @floor(double)

; CHECK: declare float @floorf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @floorf(float)

; CHECK: declare x86_fp80 @floorl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @floorl(x86_fp80)

; CHECK: declare i32 @fls(i32)
declare i32 @fls(i32)

; CHECK: declare i32 @flsl(i64)
declare i32 @flsl(i64)

; CHECK: declare i32 @flsll(i64)
declare i32 @flsll(i64)

; CHECK: declare double @fmax(double, double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @fmax(double, double)

; CHECK: declare float @fmaxf(float, float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @fmaxf(float, float)

; CHECK: declare x86_fp80 @fmaxl(x86_fp80, x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)

; CHECK: declare double @fmin(double, double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @fmin(double, double)

; CHECK: declare float @fminf(float, float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @fminf(float, float)

; CHECK: declare x86_fp80 @fminl(x86_fp80, x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @fminl(x86_fp80, x86_fp80)

; CHECK: declare double @fmod(double, double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @fmod(double, double)

; CHECK: declare float @fmodf(float, float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @fmodf(float, float)

; CHECK: declare x86_fp80 @fmodl(x86_fp80, x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @fmodl(x86_fp80, x86_fp80)

; CHECK: declare noalias noundef ptr @fopen(ptr nocapture noundef readonly, ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare ptr @fopen(ptr, ptr)

; CHECK: declare noundef i32 @fprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) [[NOFREE_NOUNWIND]]
declare i32 @fprintf(ptr, ptr, ...)

; CHECK: declare noundef i32 @fputc(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fputc(i32, ptr)

; CHECK: declare noundef i32 @fputs(ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fputs(ptr, ptr)

; CHECK: declare noundef i64 @fread(ptr nocapture noundef, i64 noundef, i64 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i64 @fread(ptr, i64, i64, ptr)

; CHECK: declare void @free(ptr allocptr nocapture noundef) [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_ALLOCKIND_FREE_FAMILY_MALLOC:#[0-9]+]]
declare void @free(ptr)

; CHECK-AIX: declare void @vec_free(ptr allocptr nocapture noundef) [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_FAMILY_VEC_MALLOC:#[0-9]+]]
declare void @vec_free(ptr)

; CHECK: declare double @frexp(double, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN_WRITEONLY:#[0-9]+]]
declare double @frexp(double, ptr)

; CHECK: declare float @frexpf(float, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @frexpf(float, ptr)

; CHECK: declare x86_fp80 @frexpl(x86_fp80, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @frexpl(x86_fp80, ptr)

; CHECK: declare noundef i32 @fscanf(ptr nocapture noundef, ptr nocapture noundef readonly, ...) [[NOFREE_NOUNWIND]]
declare i32 @fscanf(ptr, ptr, ...)

; CHECK: declare noundef i32 @fseek(ptr nocapture noundef, i64 noundef, i32 noundef) [[NOFREE_NOUNWIND]]
declare i32 @fseek(ptr, i64, i32)

; CHECK: declare noundef i32 @fseeko(ptr nocapture noundef, i64 noundef, i32 noundef) [[NOFREE_NOUNWIND]]
declare i32 @fseeko(ptr, i64, i32)

; CHECK-LINUX: declare noundef i32 @fseeko64(ptr nocapture noundef, i64 noundef, i32 noundef) [[NOFREE_NOUNWIND]]
declare i32 @fseeko64(ptr, i64, i32)

; CHECK: declare noundef i32 @fsetpos(ptr nocapture noundef, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @fsetpos(ptr, ptr)

; CHECK: declare noundef i32 @fstat(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fstat(i32, ptr)

; CHECK-LINUX: declare noundef i32 @fstat64(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fstat64(i32, ptr)

; CHECK: declare noundef i32 @fstatvfs(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fstatvfs(i32, ptr)

; CHECK-LINUX: declare noundef i32 @fstatvfs64(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @fstatvfs64(i32, ptr)

; CHECK: declare noundef i64 @ftell(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i64 @ftell(ptr)

; CHECK: declare noundef i64 @ftello(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i64 @ftello(ptr)

; CHECK-LINUX: declare noundef i64 @ftello64(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i64 @ftello64(ptr)

; CHECK: declare noundef i32 @ftrylockfile(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @ftrylockfile(ptr)

; CHECK: declare void @funlockfile(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare void @funlockfile(ptr)

; CHECK: declare noundef i64 @fwrite(ptr nocapture noundef, i64 noundef, i64 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i64 @fwrite(ptr, i64, i64, ptr)

; CHECK: declare noundef i32 @getc(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @getc(ptr)

; CHECK-KNOWN: declare noundef i32 @getc_unlocked(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
; CHECK-UNKNOWN: declare i32 @getc_unlocked(ptr){{$}}
declare i32 @getc_unlocked(ptr)

; CHECK: declare noundef i32 @getchar() [[NOFREE_NOUNWIND]]
declare i32 @getchar()

; CHECK-KNOWN: declare noundef i32 @getchar_unlocked() [[NOFREE_NOUNWIND]]
; CHECK-UNKNOWN: declare i32 @getchar_unlocked(){{$}}
declare i32 @getchar_unlocked()

; CHECK: declare noundef ptr @getenv(ptr nocapture noundef) [[NOFREE_NOUNWIND_READONLY]]
declare ptr @getenv(ptr)

; CHECK: declare noundef i32 @getitimer(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @getitimer(i32, ptr)

; CHECK: declare noundef i32 @getlogin_r(ptr nocapture noundef, i64 noundef) [[NOFREE_NOUNWIND]]
declare i32 @getlogin_r(ptr, i64)

; CHECK: declare noundef ptr @getpwnam(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare ptr @getpwnam(ptr)

; CHECK: declare noundef ptr @gets(ptr noundef) [[NOFREE_NOUNWIND]]
declare ptr @gets(ptr)

; CHECK: declare noundef i32 @gettimeofday(ptr nocapture noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @gettimeofday(ptr, ptr)

; CHECK: declare i32 @isascii(i32) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare i32 @isascii(i32)

; CHECK: declare i32 @isdigit(i32) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare i32 @isdigit(i32)

; CHECK: declare i64 @labs(i64) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare i64 @labs(i64)

; CHECK: declare noundef i32 @lchown(ptr nocapture noundef readonly, i32 noundef, i32 noundef) [[NOFREE_NOUNWIND]]
declare i32 @lchown(ptr, i32, i32)

; CHECK: declare double @ldexp(double, i32) [[NOFREE_WILLRETURN:#[0-9]+]]
declare double @ldexp(double, i32)

; CHECK: declare float @ldexpf(float, i32) [[NOFREE_WILLRETURN]]
declare float @ldexpf(float, i32)

; CHECK: declare x86_fp80 @ldexpl(x86_fp80, i32) [[NOFREE_WILLRETURN]]
declare x86_fp80 @ldexpl(x86_fp80, i32)

; CHECK: declare i64 @llabs(i64) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare i64 @llabs(i64)

; CHECK: declare double @log(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @log(double)

; CHECK: declare double @log10(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @log10(double)

; CHECK: declare float @log10f(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @log10f(float)

; CHECK: declare x86_fp80 @log10l(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @log10l(x86_fp80)

; CHECK: declare double @log1p(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @log1p(double)

; CHECK: declare float @log1pf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @log1pf(float)

; CHECK: declare x86_fp80 @log1pl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @log1pl(x86_fp80)

; CHECK: declare double @log2(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @log2(double)

; CHECK: declare float @log2f(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @log2f(float)

; CHECK: declare x86_fp80 @log2l(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @log2l(x86_fp80)

; CHECK: declare double @logb(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @logb(double)

; CHECK: declare float @logbf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @logbf(float)

; CHECK: declare x86_fp80 @logbl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @logbl(x86_fp80)

; CHECK: declare float @logf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @logf(float)

; CHECK: declare x86_fp80 @logl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @logl(x86_fp80)

; CHECK: declare noundef i32 @lstat(ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @lstat(ptr, ptr)

; CHECK-LINUX: declare noundef i32 @lstat64(ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @lstat64(ptr, ptr)

; CHECK: declare noalias noundef ptr @malloc(i64 noundef) [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCKIND_ALLOCUNINIT_ALLOCSIZE0_FAMILY_MALLOC:#[0-9]+]]
declare ptr @malloc(i64)

; CHECK-AIX: declare noalias noundef ptr @vec_malloc(i64 noundef) [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCSIZE0_FAMILY_VEC_MALLOC:#[0-9]+]]
declare ptr @vec_malloc(i64)

; CHECK-LINUX: declare noalias noundef ptr @memalign(i64 allocalign, i64) [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN:#[0-9]+]]
declare ptr @memalign(i64, i64)

; CHECK: declare ptr @memccpy(ptr noalias writeonly, ptr noalias nocapture readonly, i32, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @memccpy(ptr, ptr, i32, i64)

; CHECK-LINUX:   declare ptr @memchr(ptr, i32, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
; CHECK-DARWIN:  declare ptr @memchr(ptr, i32, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY:#[0-9]+]]
; CHECK-UNKNOWN: declare ptr @memchr(ptr, i32, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY:#[0-9]+]]
declare ptr @memchr(ptr, i32, i64)

; CHECK: declare i32 @memcmp(ptr nocapture, ptr nocapture, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare i32 @memcmp(ptr, ptr, i64)

; CHECK: declare ptr @memcpy(ptr noalias returned writeonly, ptr noalias nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @memcpy(ptr, ptr, i64)

; CHECK: declare ptr @__memcpy_chk(ptr noalias writeonly, ptr noalias nocapture readonly, i64, i64) [[ARGMEMONLY_NOFREE_NOUNWIND:#[0-9]+]]
declare ptr @__memcpy_chk(ptr, ptr, i64, i64)

; CHECK: declare ptr @mempcpy(ptr noalias writeonly, ptr noalias nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @mempcpy(ptr, ptr, i64)

; CHECK: declare ptr @memmove(ptr returned writeonly, ptr nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @memmove(ptr, ptr, i64)

; CHECK: declare ptr @memset(ptr writeonly, i32, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN:#[0-9]+]]
declare ptr @memset(ptr, i32, i64)

; CHECK: declare ptr @__memset_chk(ptr writeonly, i32, i64, i64) [[ARGMEMONLY_NOFREE_NOUNWIND]]
declare ptr @__memset_chk(ptr, i32, i64, i64)

; CHECK: declare noundef i32 @mkdir(ptr nocapture noundef readonly, i16 noundef zeroext) [[NOFREE_NOUNWIND]]
declare i32 @mkdir(ptr, i16 zeroext)

; CHECK: declare noundef i64 @mktime(ptr nocapture noundef) [[NOFREE_NOUNWIND_WILLRETURN:#[0-9]+]]
declare i64 @mktime(ptr)

; CHECK: declare double @modf(double, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @modf(double, ptr)

; CHECK: declare float @modff(float, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @modff(float, ptr)

; CHECK: declare x86_fp80 @modfl(x86_fp80, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @modfl(x86_fp80, ptr)

; CHECK: declare double @nearbyint(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @nearbyint(double)

; CHECK: declare float @nearbyintf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @nearbyintf(float)

; CHECK: declare x86_fp80 @nearbyintl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @nearbyintl(x86_fp80)

; CHECK-LINUX: declare noundef i32 @open(ptr nocapture noundef readonly, i32 noundef, ...) [[NOFREE]]
; CHECK-OPEN: declare noundef i32 @open(ptr nocapture noundef readonly, i32 noundef, ...) [[NOFREE:#[0-9]+]]
declare i32 @open(ptr, i32, ...)

; CHECK-LINUX: declare noundef i32 @open64(ptr nocapture noundef readonly, i32 noundef, ...) [[NOFREE]]
declare i32 @open64(ptr, i32, ...)

; CHECK: declare noalias noundef ptr @opendir(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare ptr @opendir(ptr)

; CHECK: declare noundef i32 @pclose(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @pclose(ptr)

; CHECK: declare void @perror(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare void @perror(ptr)

; CHECK: declare noalias noundef ptr @popen(ptr nocapture noundef readonly, ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare ptr @popen(ptr, ptr)

; CHECK: declare i32 @posix_memalign(ptr, i64, i64) [[NOFREE]]
declare i32 @posix_memalign(ptr, i64, i64)

; CHECK: declare double @pow(double, double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @pow(double, double)

; CHECK: declare float @powf(float, float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @powf(float, float)

; CHECK: declare x86_fp80 @powl(x86_fp80, x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @powl(x86_fp80, x86_fp80)

; CHECK: declare noundef i64 @pread(i32 noundef, ptr nocapture noundef, i64 noundef, i64 noundef) [[NOFREE]]
declare i64 @pread(i32, ptr, i64, i64)

; CHECK: declare noundef i32 @printf(ptr nocapture noundef readonly, ...) [[NOFREE_NOUNWIND]]
declare i32 @printf(ptr, ...)

; CHECK: declare noundef i32 @putc(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @putc(i32, ptr)

; CHECK: declare noundef i32 @putchar(i32 noundef) [[NOFREE_NOUNWIND]]
declare i32 @putchar(i32)

; CHECK-KNOWN: declare noundef i32 @putchar_unlocked(i32 noundef) [[NOFREE_NOUNWIND]]
; CHECK-UNKNOWN: declare i32 @putchar_unlocked(i32){{$}}
declare i32 @putchar_unlocked(i32)

; CHECK: declare noundef i32 @puts(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @puts(ptr)

; CHECK: declare noundef i64 @pwrite(i32 noundef, ptr nocapture noundef readonly, i64 noundef, i64 noundef) [[NOFREE]]
declare i64 @pwrite(i32, ptr, i64, i64)

; CHECK: declare void @qsort(ptr noundef, i64 noundef, i64 noundef, ptr nocapture noundef) [[NOFREE]]
declare void @qsort(ptr, i64, i64, ptr)

; CHECK: declare noundef i64 @read(i32 noundef, ptr nocapture noundef, i64 noundef) [[NOFREE]]
declare i64 @read(i32, ptr, i64)

; CHECK: declare noundef i64 @readlink(ptr nocapture noundef readonly, ptr nocapture noundef, i64 noundef) [[NOFREE_NOUNWIND]]
declare i64 @readlink(ptr, ptr, i64)

; CHECK: declare noalias noundef ptr @realloc(ptr allocptr nocapture, i64 noundef) [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_ALLOCKIND_REALLOC_ALLOCSIZE1_FAMILY_MALLOC:#[0-9]+]]
declare ptr @realloc(ptr, i64)

; CHECK: declare noalias noundef ptr @reallocf(ptr allocptr nocapture, i64 noundef) [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_ALLOCKIND_REALLOC_ALLOCSIZE1_FAMILY_MALLOC]]
declare ptr @reallocf(ptr, i64)

; CHECK-AIX: declare noalias noundef ptr @vec_realloc(ptr allocptr nocapture, i64 noundef) [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_ALLOCSIZE_FAMILY_VEC_MALLOC:#[0-9]+]]
declare ptr @vec_realloc(ptr, i64)

; CHECK: declare noundef ptr @realpath(ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare ptr @realpath(ptr, ptr)

; CHECK: declare noundef i32 @remove(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @remove(ptr)

; CHECK: declare noundef i32 @rename(ptr nocapture noundef readonly, ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @rename(ptr, ptr)

; CHECK: declare void @rewind(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare void @rewind(ptr)

; CHECK: declare double @rint(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @rint(double)

; CHECK: declare float @rintf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @rintf(float)

; CHECK: declare x86_fp80 @rintl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @rintl(x86_fp80)

; CHECK: declare noundef i32 @rmdir(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @rmdir(ptr)

; CHECK: declare double @round(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @round(double)

; CHECK: declare float @roundf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @roundf(float)

; CHECK: declare x86_fp80 @roundl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @roundl(x86_fp80)

; CHECK: declare noundef i32 @scanf(ptr nocapture noundef readonly, ...) [[NOFREE_NOUNWIND]]
declare i32 @scanf(ptr, ...)

; CHECK: declare void @setbuf(ptr nocapture noundef, ptr noundef) [[NOFREE_NOUNWIND]]
declare void @setbuf(ptr, ptr)

; CHECK: declare noundef i32 @setitimer(i32 noundef, ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i32 @setitimer(i32, ptr, ptr)

; CHECK: declare noundef i32 @setvbuf(ptr nocapture noundef, ptr noundef, i32 noundef, i64 noundef) [[NOFREE_NOUNWIND]]
declare i32 @setvbuf(ptr, ptr, i32, i64)

; CHECK: declare double @sin(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @sin(double)

; CHECK: declare float @sinf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @sinf(float)

; CHECK: declare double @sinh(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @sinh(double)

; CHECK: declare float @sinhf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @sinhf(float)

; CHECK: declare x86_fp80 @sinhl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @sinhl(x86_fp80)

; CHECK: declare x86_fp80 @sinl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @sinl(x86_fp80)

; CHECK: declare noundef i32 @snprintf(ptr noalias nocapture noundef writeonly, i64 noundef, ptr nocapture noundef readonly, ...) [[NOFREE_NOUNWIND]]
declare i32 @snprintf(ptr, i64, ptr, ...)

; CHECK: declare noundef i32 @sprintf(ptr noalias nocapture noundef writeonly, ptr nocapture noundef readonly, ...) [[NOFREE_NOUNWIND]]
declare i32 @sprintf(ptr, ptr, ...)

; CHECK: declare double @sqrt(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @sqrt(double)

; CHECK: declare float @sqrtf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @sqrtf(float)

; CHECK: declare x86_fp80 @sqrtl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @sqrtl(x86_fp80)

; CHECK: declare noundef i32 @sscanf(ptr nocapture noundef readonly, ptr nocapture noundef readonly, ...) [[NOFREE_NOUNWIND]]
declare i32 @sscanf(ptr, ptr, ...)

; CHECK: declare noundef i32 @stat(ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @stat(ptr, ptr)

; CHECK-LINUX: declare noundef i32 @stat64(ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @stat64(ptr, ptr)

; CHECK: declare noundef i32 @statvfs(ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @statvfs(ptr, ptr)

; CHECK-LINUX: declare noundef i32 @statvfs64(ptr nocapture noundef readonly, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @statvfs64(ptr, ptr)

; CHECK: declare ptr @stpcpy(ptr noalias writeonly, ptr noalias nocapture readonly) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @stpcpy(ptr, ptr)

; CHECK: declare ptr @stpncpy(ptr noalias writeonly, ptr noalias nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @stpncpy(ptr, ptr, i64)

; CHECK: declare i32 @strcasecmp(ptr nocapture, ptr nocapture) [[NOFREE_NOUNWIND_READONLY_WILLRETURN:#[0-9]+]]
declare i32 @strcasecmp(ptr, ptr)

; CHECK: declare ptr @strcat(ptr noalias returned, ptr noalias nocapture readonly) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @strcat(ptr, ptr)

; CHECK: declare ptr @strchr(ptr, i32) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare ptr @strchr(ptr, i32)

; CHECK: declare i32 @strcmp(ptr nocapture, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare i32 @strcmp(ptr, ptr)

; CHECK: declare i32 @strcoll(ptr nocapture, ptr nocapture) [[NOFREE_NOUNWIND_READONLY_WILLRETURN]]
declare i32 @strcoll(ptr, ptr)

; CHECK: declare ptr @strcpy(ptr noalias returned writeonly, ptr noalias nocapture readonly) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @strcpy(ptr, ptr)

; CHECK: declare i64 @strcspn(ptr nocapture, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare i64 @strcspn(ptr, ptr)

; CHECK: declare noalias ptr @strdup(ptr nocapture readonly) [[INACCESSIBLEMEMORARGONLY_NOFREE_NOUNWIND_WILLRETURN_FAMILY_MALLOC:#[0-9]+]]
declare ptr @strdup(ptr)

; CHECK: declare i64 @strlen(ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare i64 @strlen(ptr)

; CHECK: declare i32 @strncasecmp(ptr nocapture, ptr nocapture, i64) [[NOFREE_NOUNWIND_READONLY_WILLRETURN]]
declare i32 @strncasecmp(ptr, ptr, i64)

; CHECK: declare ptr @strncat(ptr noalias returned, ptr noalias nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @strncat(ptr, ptr, i64)

; CHECK: declare i32 @strncmp(ptr nocapture, ptr nocapture, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare i32 @strncmp(ptr, ptr, i64)

; CHECK: declare ptr @strncpy(ptr noalias returned writeonly, ptr noalias nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @strncpy(ptr, ptr, i64)

; CHECK: declare noalias ptr @strndup(ptr nocapture readonly, i64 noundef) [[INACCESSIBLEMEMORARGONLY_NOFREE_NOUNWIND_WILLRETURN_FAMILY_MALLOC]]
declare ptr @strndup(ptr, i64)

; CHECK: declare i64 @strnlen(ptr nocapture, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY_WILLRETURN:#[0-9]+]]
declare i64 @strnlen(ptr, i64)

; CHECK: declare ptr @strpbrk(ptr, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY_WILLRETURN]]
declare ptr @strpbrk(ptr, ptr)

; CHECK: declare ptr @strrchr(ptr, i32) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare ptr @strrchr(ptr, i32)

; CHECK: declare i64 @strspn(ptr nocapture, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY]]
declare i64 @strspn(ptr, ptr)

; CHECK: declare ptr @strstr(ptr, ptr nocapture) [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY_WILLRETURN]]
declare ptr @strstr(ptr, ptr)

; CHECK: declare double @strtod(ptr readonly, ptr nocapture) [[NOFREE_NOUNWIND_WILLRETURN]]
declare double @strtod(ptr, ptr)

; CHECK: declare float @strtof(ptr readonly, ptr nocapture) [[NOFREE_NOUNWIND_WILLRETURN]]
declare float @strtof(ptr, ptr)

; CHECK: declare ptr @strtok(ptr, ptr nocapture readonly) [[NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @strtok(ptr, ptr)

; CHECK: declare ptr @strtok_r(ptr, ptr nocapture readonly, ptr) [[NOFREE_NOUNWIND_WILLRETURN]]
declare ptr @strtok_r(ptr, ptr, ptr)

; CHECK: declare i64 @strtol(ptr readonly, ptr nocapture, i32) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i64 @strtol(ptr, ptr, i32)

; CHECK: declare x86_fp80 @strtold(ptr readonly, ptr nocapture) [[NOFREE_NOUNWIND_WILLRETURN]]
declare x86_fp80 @strtold(ptr, ptr)

; CHECK: declare i64 @strtoll(ptr readonly, ptr nocapture, i32) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i64 @strtoll(ptr, ptr, i32)

; CHECK: declare i64 @strtoul(ptr readonly, ptr nocapture, i32) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i64 @strtoul(ptr, ptr, i32)

; CHECK: declare i64 @strtoull(ptr readonly, ptr nocapture, i32) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i64 @strtoull(ptr, ptr, i32)

; CHECK: declare i64 @strxfrm(ptr nocapture, ptr nocapture readonly, i64) [[NOFREE_NOUNWIND_WILLRETURN]]
declare i64 @strxfrm(ptr, ptr, i64)

; CHECK: declare noundef i32 @system(ptr nocapture noundef readonly) [[NOFREE]]
declare i32 @system(ptr)

; CHECK: declare double @tan(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @tan(double)

; CHECK: declare float @tanf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @tanf(float)

; CHECK: declare double @tanh(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @tanh(double)

; CHECK: declare float @tanhf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @tanhf(float)

; CHECK: declare x86_fp80 @tanhl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @tanhl(x86_fp80)

; CHECK: declare x86_fp80 @tanl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @tanl(x86_fp80)

; CHECK: declare noundef i64 @times(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i64 @times(ptr)

; CHECK: declare noalias noundef ptr @tmpfile() [[NOFREE_NOUNWIND]]
declare ptr @tmpfile()

; CHECK-LINUX: declare noalias noundef ptr @tmpfile64() [[NOFREE_NOUNWIND]]
declare ptr @tmpfile64()

; CHECK: declare i32 @toascii(i32) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare i32 @toascii(i32)

; CHECK: declare double @trunc(double) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare double @trunc(double)

; CHECK: declare float @truncf(float) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare float @truncf(float)

; CHECK: declare x86_fp80 @truncl(x86_fp80) [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]]
declare x86_fp80 @truncl(x86_fp80)

; CHECK: declare noundef i32 @uname(ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @uname(ptr)

; CHECK: declare noundef i32 @ungetc(i32 noundef, ptr nocapture noundef) [[NOFREE_NOUNWIND]]
declare i32 @ungetc(i32, ptr)

; CHECK: declare noundef i32 @unlink(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @unlink(ptr)

; CHECK: declare noundef i32 @unsetenv(ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @unsetenv(ptr)

; CHECK: declare noundef i32 @utime(ptr nocapture noundef readonly, ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @utime(ptr, ptr)

; CHECK: declare noundef i32 @utimes(ptr nocapture noundef readonly, ptr nocapture noundef readonly) [[NOFREE_NOUNWIND]]
declare i32 @utimes(ptr, ptr)

; CHECK: declare noalias noundef ptr @valloc(i64 noundef) [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCKIND_ALLOCUNINIT_ALLOCSIZE0_FAMILY_MALLOC]]
declare ptr @valloc(i64)

; CHECK: declare noundef i32 @vfprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @vfprintf(ptr, ptr, ptr)

; CHECK: declare noundef i32 @vfscanf(ptr nocapture noundef, ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @vfscanf(ptr, ptr, ptr)

; CHECK: declare noundef i32 @vprintf(ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @vprintf(ptr, ptr)

; CHECK: declare noundef i32 @vscanf(ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @vscanf(ptr, ptr)

; CHECK: declare noundef i32 @vsnprintf(ptr nocapture noundef, i64 noundef, ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @vsnprintf(ptr, i64, ptr, ptr)

; CHECK: declare noundef i32 @vsprintf(ptr nocapture noundef, ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @vsprintf(ptr, ptr, ptr)

; CHECK: declare noundef i32 @vsscanf(ptr nocapture noundef readonly, ptr nocapture noundef readonly, ptr noundef) [[NOFREE_NOUNWIND]]
declare i32 @vsscanf(ptr, ptr, ptr)

; CHECK: declare noundef i64 @write(i32 noundef, ptr nocapture noundef readonly, i64 noundef) [[NOFREE]]
declare i64 @write(i32, ptr, i64)


; memset_pattern{4,8,16} aren't available everywhere.
; CHECK-DARWIN: declare void @memset_pattern4(ptr nocapture writeonly, ptr nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare void @memset_pattern4(ptr, ptr, i64)
; CHECK-DARWIN: declare void @memset_pattern8(ptr nocapture writeonly, ptr nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare void @memset_pattern8(ptr, ptr, i64)
; CHECK-DARWIN: declare void @memset_pattern16(ptr nocapture writeonly, ptr nocapture readonly, i64) [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]]
declare void @memset_pattern16(ptr, ptr, i64)

; CHECK-DAG: attributes [[NOFREE_NOUNWIND_WILLRETURN]] = { mustprogress nofree nounwind willreturn }
; CHECK-DAG: attributes [[NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]] = { mustprogress nofree nounwind willreturn memory(write) }
; CHECK-DAG: attributes [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN_WRITEONLY]] = { mustprogress nofree nounwind willreturn memory(argmem: write) }
; CHECK-DAG: attributes [[NOFREE_NOUNWIND]] = { nofree nounwind }
; CHECK-DAG: attributes [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCKIND_ALLOCUNINIT_ALLOCSIZE1_FAMILY_MALLOC]] = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized,aligned") allocsize(1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK-DAG: attributes [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCKIND_ALLOCZEROED_ALLOCSIZE01_FAMILY_MALLOC]] = { mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK-DAG: attributes [[NOFREE_NOUNWIND_READONLY_WILLRETURN]] = { mustprogress nofree nounwind willreturn memory(read) }
; CHECK-DAG: attributes [[ARGMEMONLY_NOFREE_NOUNWIND_WILLRETURN]] = { mustprogress nofree nounwind willreturn memory(argmem: readwrite) }
; CHECK-DAG: attributes [[NOFREE_NOUNWIND_READONLY]] = { nofree nounwind memory(read) }
; CHECK-DAG: attributes [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_ALLOCKIND_FREE_FAMILY_MALLOC]] = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK-DAG: attributes [[NOFREE_WILLRETURN]] = { mustprogress nofree willreturn }
; CHECK-DAG: attributes [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCKIND_ALLOCUNINIT_ALLOCSIZE0_FAMILY_MALLOC]] = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK-DAG: attributes [[ARGMEMONLY_NOFREE_NOUNWIND_READONLY_WILLRETURN]] = { mustprogress nofree nounwind willreturn memory(argmem: read) }
; CHECK-DAG: attributes [[NOFREE]] = { nofree }
; CHECK-DAG: attributes [[ARGMEMONLY_NOFREE_NOUNWIND]] = { nofree nounwind memory(argmem: readwrite) }
; CHECK-DAG: attributes [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_ALLOCKIND_REALLOC_ALLOCSIZE1_FAMILY_MALLOC]] = { mustprogress nounwind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }
; CHECK-DAG: attributes [[INACCESSIBLEMEMORARGONLY_NOFREE_NOUNWIND_WILLRETURN_FAMILY_MALLOC]] = { mustprogress nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" }

; CHECK-NVPTX-DAG: attributes [[NOFREE_NOUNWIND_READNONE]] = { nofree nosync nounwind memory(none) }

; CHECK-AIX-DAG: attributes [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCSIZE0_FAMILY_VEC_MALLOC]] = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="vec_malloc" }
; CHECK-AIX-DAG: attributes [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_FAMILY_VEC_MALLOC]] = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="vec_malloc" }
; CHECK-AIX-DAG: attributes [[INACCESSIBLEMEMORARGMEMONLY_NOUNWIND_WILLRETURN_ALLOCSIZE_FAMILY_VEC_MALLOC]] = { mustprogress nounwind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="vec_malloc" }
; CHECK-AIX-DAG: attributes [[INACCESSIBLEMEMONLY_NOFREE_NOUNWIND_WILLRETURN_ALLOCSIZE01_FAMILY_VEC_MALLOC]] = { mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="vec_malloc" }
