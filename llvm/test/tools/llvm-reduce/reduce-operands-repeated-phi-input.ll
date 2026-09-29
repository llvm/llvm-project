; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK %s < %t

; Make sure if we're replacing the value in a phi, it's replaced for
; all repeats of the same incoming block.

; CHECK-INTERESTINGNESS: switch
; CHECK-INTERESTINGNESS: phi
; CHECK-INTERESTINGNESS-SAME: [ %arg1, %bb1 ]
; CHECK-INTERESTINGNESS: phi
; CHECK-INTERESTINGNESS-SAME: [ %arg3, %bb1 ]
; CHECK-INTERESTINGNESS: store volatile i32 %
; CHECK-INTERESTINGNESS: store volatile float %

; CHECK: %phi.i32 = phi i32 [ 0, %entry ], [ 0, %entry ], [ %arg1, %bb1 ]
; CHECK: %phi.f32 = phi float [ 0.000000e+00, %entry ], [ 0.000000e+00, %entry ], [ %arg3, %bb1 ]
define void @foo(i32 %arg0, i32 %arg1, float %arg2, float %arg3) {
entry:
  switch i32 %arg0, label %ret [
    i32 3, label %bb1
    i32 4, label %bb2
    i32 12, label %bb2
  ]

bb1:
  br label %bb2

bb2:
  %phi.i32 = phi i32 [ %arg0, %entry ], [ %arg0, %entry ], [ %arg1, %bb1 ]
  %phi.f32 = phi float [ %arg2, %entry ], [ %arg2, %entry ], [ %arg3, %bb1 ]
  store volatile i32 %phi.i32, ptr undef
  store volatile float %phi.f32, ptr undef
  br label %ret

ret:
  ret void
}
