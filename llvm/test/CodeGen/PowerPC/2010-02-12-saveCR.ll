; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin -mcpu=g4 -break-anti-dependencies=none | FileCheck %s
; ModuleID = 'hh.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"
target triple = "powerpc-apple-darwin9.6"

define void @foo() nounwind {
entry:
; Note that part of what is being checked here is proper register reuse.
; CHECK: mfcr [[T1:r[0-9]+]]                         ; cr2
; CHECK: lis [[T2:r[0-9]+]], 1
; CHECK: addi r3, r1, 72
; CHECK: rotlwi [[T1]], [[T1]], 8
; CHECK: ori [[T2]], [[T2]], 34540
; CHECK: stwx [[T1]], r1, [[T2]]
; CHECK: lis [[T3:r[0-9]+]], 1
; CHECK: mfcr [[T4:r[0-9]+]]                         ; cr3
; CHECK: ori [[T3]], [[T3]], 34536
; CHECK: rotlwi [[T4]], [[T4]], 12
; CHECK: stwx [[T4]], r1, [[T3]]
  %x = alloca [100000 x i8]                       ; <[100000 x i8]*> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %x1 = bitcast [100000 x i8]* %x to i8*          ; <i8*> [#uses=1]
  call void @bar(i8* %x1) nounwind
  call void asm sideeffect "", "~{cr2},~{cr3}"() nounwind
  br label %return

return:                                           ; preds = %entry
; CHECK: lis [[T1:r[0-9]+]], 1
; CHECK: ori [[T1]], [[T1]], 34536
; CHECK: lwzx [[T1]], r1, [[T1]]
; CHECK: rotlwi [[T1]], [[T1]], 20
; CHECK: mtcrf 16, [[T1]]
; CHECK: lis [[T1]], 1
; CHECK: ori [[T1]], [[T1]], 34540
; CHECK: lwzx [[T1]], r1, [[T1]]
; CHECK: rotlwi [[T1]], [[T1]], 24
; CHECK: mtcrf 32, [[T1]]
  ret void
}

declare void @bar(i8*)
