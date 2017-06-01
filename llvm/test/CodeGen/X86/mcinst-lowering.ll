; RUN: llc --show-mc-encoding < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

declare i32 @foo();

define i32 @f0(i32* nocapture %x) nounwind readonly ssp {
entry:
  %tmp1 = call i32 @foo()
; CHECK: cmpl $16777216, %eax
; CHECK: # encoding: [0x3d,0x00,0x00,0x00,0x01]
  %cmp = icmp eq i32 %tmp1, 16777216              ; <i1> [#uses=1]

  %conv = zext i1 %cmp to i32                     ; <i32> [#uses=1]
  ret i32 %conv
}

define i32 @f1() nounwind {
  %ax = tail call i16 asm sideeffect "", "={ax},~{dirflag},~{fpsr},~{flags}"()
  %conv = sext i16 %ax to i32
  ret i32 %conv

; CHECK-LABEL: f1:
; CHECK: cwtl ## encoding: [0x98]
}

define i64 @f2() nounwind {
  %eax = tail call i32 asm sideeffect "", "={ax},~{dirflag},~{fpsr},~{flags}"()
  %conv = sext i32 %eax to i64
  ret i64 %conv

; CHECK-LABEL: f2:
; CHECK: cltq ## encoding: [0x48,0x98]
}
