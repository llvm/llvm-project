; RUN: opt < %s -passes=vector-combine -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Issue #213423
; Check that we do not crash when multiple extracts share the same index instruction
; and it needs to be frozen.
define i32 @test_shared_index(ptr %p, i32 %idx.base) {
; CHECK-LABEL: @test_shared_index(
; CHECK:       [[FROZEN:%.*]] = freeze i32 %idx.base
; CHECK-NEXT:  %idx = and i32 [[FROZEN]], 1
; CHECK-DAG:   [[GEP1:%.*]] = getelementptr inbounds <2 x i32>, ptr %p, i32 0, i32 %idx
; CHECK-DAG:   %e1.scalar = load i32, ptr [[GEP1]], align 4
; CHECK-DAG:   [[GEP2:%.*]] = getelementptr inbounds <2 x i32>, ptr %p, i32 0, i32 %idx
; CHECK-DAG:   %e2.scalar = load i32, ptr [[GEP2]], align 4
; CHECK:       %add = add i32 %e1.scalar, %e2.scalar
; CHECK-NEXT:  ret i32 %add
;
  %idx = and i32 %idx.base, 1
  %v = load <2 x i32>, ptr %p
  %e1 = extractelement <2 x i32> %v, i32 %idx
  %e2 = extractelement <2 x i32> %v, i32 %idx
  %add = add i32 %e1, %e2
  ret i32 %add
}
