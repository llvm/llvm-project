; RUN: llc < %s -mtriple=powerpc-apple-darwin -mcpu=g4 | FileCheck %s
; ModuleID = 'hh.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"
target triple = "powerpc-apple-darwin9.6"
; This formerly used R0 for both the stack address and CR.

define void @foo() nounwind {
entry:
;CHECK:  mfcr r2
;CHECK:  lis r3, 1
;CHECK:  rlwinm r2, r2, 8, 0, 31
;CHECK:  ori r3, r3, 34524
;CHECK:  stwx r2, r1, r3
; Make sure that the register scavenger returns the same temporary register.
;CHECK:  mfcr r2
;CHECK:  lis r3, 1
;CHECK:  rlwinm r2, r2, 12, 0, 31
;CHECK:  ori r3, r3, 34520
;CHECK:  stwx r2, r1, r3
  %x = alloca [100000 x i8]                       ; <[100000 x i8]*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %x1 = bitcast [100000 x i8]* %x to i8*          ; <i8*> [#uses=1]
  call void @bar(i8* %x1) nounwind
  call void asm sideeffect "", "~{cr2},~{cr3}"() nounwind
  br label %return

return:                                           ; preds = %entry
;CHECK:  lis r3, 1
;CHECK:  ori r3, r3, 34524
;CHECK:  lwzx r2, r1, r3
;CHECK:  rlwinm r2, r2, 24, 0, 31
;CHECK:  mtcrf 32, r2
  ret void
}

declare void @bar(i8*)
