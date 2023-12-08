; RUN: llc -verify-machineinstrs -code-model=small < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; This test check if the TOC entry symbol name won't clash with global .LC0
; and .LC2 symbols defined in the module.

@.LC0 = internal global [5 x i8] c".LC0\00"
@.LC2 = internal global [5 x i8] c".LC2\00"

define i32 @foo(double %X, double %Y) nounwind readnone {
  ; The 1.0 and 3.0 constants generate two TOC entries
  %cmp = fcmp oeq double %X, 1.000000e+00
  %conv = zext i1 %cmp to i32
  %cmp1 = fcmp oeq double %Y, 3.000000e+00
  %conv2 = zext i1 %cmp1 to i32
  %add = add nsw i32 %conv2, %conv
  ret i32 %add
}

; Check the creation of 2 .tc entries for both double constants. They
; avoid name clash with global constants .LC0 and .LC2
; CHECK: .section	.toc,"aw",@progbits
; CHECK: .LC{{.*}}:
; CHECK-NEXT: .tc {{[\._a-zA-Z0-9]+}}[TC],{{[\._a-zA-Z0-9]+}}
; CHECK: .LC{{.*}}:
; CHECK-NEXT: .tc {{[\._a-zA-Z0-9]+}}[TC],{{[\._a-zA-Z0-9]+}}
