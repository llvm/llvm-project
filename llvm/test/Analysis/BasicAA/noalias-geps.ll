; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

; Check that geps with equal base offsets of noalias base pointers stay noalias.
define i32 @test(i32* %p, i16 %i) {
; CHECK-LABEL: Function: test:
  %pi = getelementptr i32, i32* %p, i32 0
  %pi.next = getelementptr i32, i32* %p, i32 1
  %b = icmp eq i16 %i, 0
  br i1 %b, label %bb1, label %bb2

bb1:
  %f = getelementptr i32, i32* %pi, i32 1
  %g = getelementptr i32, i32* %pi.next, i32 1
  br label %bb3
bb2:
  %f2 = getelementptr i32, i32* %pi, i32 1
  %g2 = getelementptr i32, i32* %pi.next, i32 1
  br label %bb3

bb3:
  %ptr_phi = phi i32* [ %f, %bb1 ], [ %f2, %bb2 ]
  %ptr_phi2 = phi i32* [ %g, %bb1 ], [ %g2, %bb2 ]
; CHECK: NoAlias: i32* %f1, i32* %g1
  %f1 = getelementptr i32, i32* %ptr_phi , i32 1
  %g1 = getelementptr i32, i32* %ptr_phi2 , i32 1

ret i32 0
}

; Check that geps with equal indices of noalias base pointers stay noalias.
define i32 @test2([2 x i32]* %p, i32 %i) {
; CHECK-LABEL: Function: test2:
  %pi = getelementptr [2 x i32], [2 x i32]* %p, i32 0
  %pi.next = getelementptr [2 x i32], [2 x i32]* %p, i32 1
  %b = icmp eq i32 %i, 0
  br i1 %b, label %bb1, label %bb2

bb1:
  %f = getelementptr [2 x i32], [2 x i32]* %pi, i32 1
  %g = getelementptr [2 x i32], [2 x i32]* %pi.next, i32 1
  br label %bb3
bb2:
  %f2 = getelementptr [2 x i32], [2 x i32]* %pi, i32 1
  %g2 = getelementptr [2 x i32], [2 x i32]* %pi.next, i32 1
  br label %bb3
bb3:
  %ptr_phi = phi [2 x i32]* [ %f, %bb1 ], [ %f2, %bb2 ]
  %ptr_phi2 = phi [2 x i32]* [ %g, %bb1 ], [ %g2, %bb2 ]
; CHECK: NoAlias: i32* %f1, i32* %g1
  %f1 = getelementptr [2 x i32], [2 x i32]* %ptr_phi , i32 1, i32 %i
  %g1 = getelementptr [2 x i32], [2 x i32]* %ptr_phi2 , i32 1, i32 %i

ret i32 0
}
