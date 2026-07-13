; This test was return to make sure a hardware loop is not generated if a
; returning call is present in the basic block.
; RUN: llc -O2 -mtriple=hexagon < %s | FileCheck %s
; CHECK-NOT: loop
; CHECK-NOT: endloop

; Function Attrs: nounwind
define void @f0() local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v4, %b1 ], [ 2, %b0 ]
  %v1 = phi double [ %v3, %b1 ], [ 1.000000e+00, %b0 ]
  %v2 = sitofp i32 %v0 to double
  %v3 = fmul double %v2, %v1
  %v4 = add nuw nsw i32 %v0, 1
  %v5 = icmp eq i32 %v0, undef
  br i1 %v5, label %b2, label %b1

b2:                                               ; preds = %b1
  %v6 = fdiv double undef, %v3
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="-hvx,-long-calls" }
