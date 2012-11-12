; RUN: llc < %s -mcpu=generic -march=x86-64 | FileCheck %s
; Verify that we are using the efficient uitofp --> sitofp lowering illustrated
; by the compiler_rt implementation of __floatundisf.
; <rdar://problem/8493982>

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; CHECK: testq %rdi, %rdi
; CHECK-NEXT: js LBB0_1
; CHECK: cvtsi2ss
; CHECK-NEXT: ret
; CHECK: LBB0_1
; CHECK: shrq
; CHECK-NEXT: andq
; CHECK-NEXT: orq
; CHECK-NEXT: cvtsi2ss
define float @test(i64 %a) {
entry:
  %b = uitofp i64 %a to float
  ret float %b
}
