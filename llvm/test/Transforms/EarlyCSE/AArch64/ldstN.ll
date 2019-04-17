; RUN: opt -S -early-cse < %s | FileCheck %s
; RUN: opt -S -basicaa -early-cse-memssa < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

declare { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0v4i16(<4 x i16>*)

; Although the store and the ld4 are using the same pointer, the
; data can not be reused because ld4 accesses multiple elements.
define { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @foo() {
entry:
  store <4 x i16> undef, <4 x i16>* undef, align 8
  %0 = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0v4i16(<4 x i16>* undef)
  ret { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } %0
; CHECK-LABEL: @foo(
; CHECK: store
; CHECK-NEXT: call
; CHECK-NEXT: ret
}
