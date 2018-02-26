; RUN: opt < %s -cost-model -cost-kind=latency -analyze -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s --check-prefix=LATENCY
; RUN: opt < %s -cost-model -cost-kind=code-size -analyze -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s --check-prefix=CODESIZE

; Tests if the interface TargetTransformInfo::getInstructionCost() works correctly.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)

define i64 @foo(i64 %arg) {

  ; LATENCY:  cost of 0 {{.*}} alloca i32
  ; CODESIZE: cost of 0 {{.*}} alloca i32
  %A1 = alloca i32, align 8

  ; LATENCY:  cost of 1 {{.*}} alloca i64, i64 undef
  ; CODESIZE: cost of 1 {{.*}} alloca i64, i64 undef
  %A2 = alloca i64, i64 undef, align 8

  ; LATENCY:  cost of 1 {{.*}} %I64 = add
  ; CODESIZE: cost of 1 {{.*}} %I64 = add
  %I64 = add i64 undef, undef

  ; LATENCY:  cost of 4 {{.*}} load
  ; CODESIZE: cost of 1 {{.*}} load
  load i64, i64* undef, align 4

  ; LATENCY:  cost of 0 {{.*}} bitcast
  ; CODESIZE: cost of 0 {{.*}} bitcast
  %BC = bitcast i8* undef to i32*

  ; LATENCY:  cost of 0 {{.*}} inttoptr
  ; CODESIZE: cost of 0 {{.*}} inttoptr
  %I2P = inttoptr i64 undef to i8*

  ; LATENCY:  cost of 0 {{.*}} ptrtoint
  ; CODESIZE: cost of 0 {{.*}} ptrtoint
  %P2I = ptrtoint i8* undef to i64

  ; LATENCY:  cost of 0 {{.*}} trunc
  ; CODESIZE: cost of 0 {{.*}} trunc
  %TC = trunc i64 undef to i32

  ; LATENCY:  cost of 1 {{.*}} call
  ; CODESIZE: cost of 1 {{.*}} call
  %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 undef, i32 undef)

  ; LATENCY:  cost of 40 {{.*}} call void undef
  ; CODESIZE: cost of 1 {{.*}} call void undef
  call void undef()

  ; LATENCY:  cost of 1 {{.*}} ret
  ; CODESIZE: cost of 1 {{.*}} ret
  ret i64 undef
}
