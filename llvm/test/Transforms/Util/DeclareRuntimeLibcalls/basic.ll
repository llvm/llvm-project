; REQUIRES: x86-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; Check an already declared function
; CHECK: declare float @logf(float)
declare float @logf(float)

; Check an already defined function
; CHECK: define float @sinf(float %x) {
define float @sinf(float %x) {
  ret float %x
}

; CHECK: declare void @_Unwind_Resume(...)

; CHECK: declare ptr @__memcpy_chk(ptr, ptr, i64, i64)
; CHECK: declare ptr @__memmove_chk(ptr, ptr, i64, i64)
; CHECK: declare ptr @__memset_chk(ptr, i32, i64, i64)

; CHECK: declare void @__umodti3(...)

; CHECK: declare float @acosf(float)

; CHECK: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) [[CALLOC_ATTRS:#[0-9]+]]

; CHECK: declare double @fdim(double, double)
; CHECK: declare float @fdimf(float, float)
; CHECK: declare x86_fp80 @fdiml(x86_fp80, x86_fp80)

; CHECK: declare void @free(ptr allocptr noundef captures(none)) [[FREE_ATTRS:#[0-9]+]]

; CHECK: declare noalias noundef ptr @malloc(i64 noundef) [[MALLOC_ATTRS:#[0-9]+]]

; CHECK: declare double @nan(ptr)
; CHECK: declare float @nanf(ptr)
; CHECK: declare x86_fp80 @nanl(ptr)

; CHECK: declare double @nexttoward(double, x86_fp80)
; CHECK: declare float @nexttowardf(float, x86_fp80)
; CHECK: declare x86_fp80 @nexttowardl(x86_fp80, x86_fp80)

; CHECK: declare double @remainder(double, double)
; CHECK: declare float @remainderf(float, float)
; CHECK: declare x86_fp80 @remainderl(x86_fp80, x86_fp80)

; CHECK: declare double @remquo(double, double, ptr)
; CHECK: declare float @remquof(float, float, ptr)
; CHECK: declare x86_fp80 @remquol(x86_fp80, x86_fp80, ptr)

; CHECK: declare double @scalbln(double, i32)
; CHECK: declare float @scalblnf(float, i32)
; CHECK: declare x86_fp80 @scalblnl(x86_fp80, i32)

; CHECK: declare double @scalbn(double, i32)
; CHECK: declare float @scalbnf(float, i32)
; CHECK: declare x86_fp80 @scalbnl(x86_fp80, i32)

; CHECK: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]

; CHECK: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]

; CHECK: declare double @tgamma(double)
; CHECK: declare float @tgammaf(float)
; CHECK: declare x86_fp80 @tgammal(x86_fp80)

; CHECK: declare x86_fp80 @truncl(x86_fp80)

; CHECK: attributes [[CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc") allocsize(0,1) "alloc-family"="malloc" }
; CHECK: attributes [[FREE_ATTRS]] = { nounwind willreturn allockind("free") "alloc-family"="malloc" }
; CHECK: attributes [[MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) "alloc-family"="malloc" }
