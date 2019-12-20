; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -break-anti-dependencies=none | FileCheck %s
; ModuleID = 'hh.c'
;;;;; target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"
target triple = "powerpc-unknown-linux-gnu.6"

define void @foo() nounwind {
entry:
; Note that part of what is being checked here is proper register reuse.
; CHECK: mfcr [[T1:[0-9]+]]
; CHECK-DAG: subf 0, 0, 1
; CHECK-DAG: ori [[T2:[0-9]+]], [[T2]], 34492
; CHECK-DAG: stwx [[T1]], 1, [[T2]]
; CHECK-DAG: addi 3, 1, 28
; CHECK: bl bar
  %x = alloca [100000 x i8]                       ; <[100000 x i8]*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %x1 = bitcast [100000 x i8]* %x to i8*          ; <i8*> [#uses=1]
  call void @bar(i8* %x1) nounwind
  call void asm sideeffect "", "~{cr2},~{cr3}"() nounwind
  br label %return

return:                                           ; preds = %entry
; CHECK: ori [[T2]], [[T2]], 34492
; CHECK: lwzx [[T1]], 1, [[T2]]
; CHECK: mtcrf 32, [[T1]]
; CHECK: mtcrf 16, [[T1]]
  ret void
}

declare void @bar(i8*)
