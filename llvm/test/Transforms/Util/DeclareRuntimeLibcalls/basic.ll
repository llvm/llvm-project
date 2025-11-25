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

; CHECK: declare void @__memcpy_chk(...)
; CHECK: declare void @__memmove_chk(...)
; CHECK: declare void @__memset_chk(...)

; CHECK: declare void @__umodti3(...)

; CHECK: declare void @acosf(...)

; CHECK: declare void @fdim(...)
; CHECK: declare void @fdimf(...)
; CHECK: declare void @fdiml(...)

; CHECK: declare void @nan(...)
; CHECK: declare void @nanf(...)
; CHECK: declare void @nanl(...)

; CHECK: declare void @nexttoward(...)
; CHECK: declare void @nexttowardf(...)
; CHECK: declare void @nexttowardl(...)

; CHECK: declare void @remainder(...)
; CHECK: declare void @remainderf(...)
; CHECK: declare void @remainderl(...)

; CHECK: declare void @remquo(...)
; CHECK: declare void @remquof(...)
; CHECK: declare void @remquol(...)

; CHECK: declare void @scalbln(...)
; CHECK: declare void @scalblnf(...)
; CHECK: declare void @scalblnl(...)

; CHECK: declare void @scalbn(...)
; CHECK: declare void @scalbnf(...)
; CHECK: declare void @scalbnl(...)

; CHECK: declare nofpclass(ninf nsub nnorm) double @sqrt(double) [[SQRT_ATTRS:#[0-9]+]]

; CHECK: declare nofpclass(ninf nsub nnorm) float @sqrtf(float) [[SQRT_ATTRS:#[0-9]+]]

; CHECK: declare void @tgamma(...)
; CHECK: declare void @tgammaf(...)
; CHECK: declare void @tgammal(...)

; CHECK: declare void @truncl(...)

