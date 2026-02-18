; RUN: opt -passes=loop-vectorize -enable-epilogue-vectorization -force-vector-width=2 -epilogue-vectorization-force-VF=2 -S < %s | FileCheck %s
;
; Test case for issue #179407: LoopVectorize should not crash when
; vectorizing loops in functions with GC statepoint.
;
; The issue was that fixScalarResumeValuesFromBypass assumed
; IVPhi->getIncomingValueForBlock(PH) always returns a PHINode, but in
; functions with GC statepoint, it could return a constant or other value.
;
; This test verifies that epilogue vectorization works correctly with
; GC statepoint by checking that the loop is vectorized and the
; bypass blocks are properly created.

; CHECK: define void @wombat
; CHECK-SAME: gc "statepoint-example"
; CHECK: vector.body:
; CHECK: vec.epilog.vector.body:
; CHECK: vec.epilog.scalar.ph:
; CHECK: bb1:

define void @wombat(i64 %arg) gc "statepoint-example" {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i64 [ 0, %bb ], [ %add, %bb1 ]
  %phi2 = phi i32 [ 0, %bb ], [ %or, %bb1 ]
  %phi3 = phi i32 [ 0, %bb ], [ %select, %bb1 ]
  %icmp = icmp eq i32 0, 0
  %select = select i1 %icmp, i32 0, i32 %phi3
  %or = or i32 %phi2, 0
  %add = add i64 %phi, 1
  %icmp4 = icmp ult i64 %phi, %arg
  br i1 %icmp4, label %bb1, label %bb5

bb5:                                              ; preds = %bb1
  %phi6 = phi i32 [ %select, %bb1 ]
  %phi7 = phi i32 [ %or, %bb1 ]
  ret void
}
