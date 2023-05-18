; RUN: llc -march=hexagon -disable-block-placement < %s | FileCheck %s

; Check that the branch to the block b10 is marked as taken (i.e. ":t").
; CHECK-LABEL: foo
; CHECK: if ({{.*}}) jump:t .LBB0_[[LAB:[0-9]+]]
; CHECK: [[LAB]]:
; CHECK: add({{.*}},#65)

target triple = "hexagon"

define i32 @foo(i32 %a0) local_unnamed_addr #0 {
b1:
  %v2 = icmp eq i32 %a0, 0
  br i1 %v2, label %b3, label %b10, !prof !0

b3:                                               ; preds = %b1
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v5 = phi i32 [ %v6, %b4 ], [ 0, %b3 ]
  %v6 = add nuw nsw i32 %v5, 1
  %v7 = mul nuw nsw i32 %v5, 67
  %v8 = tail call i32 @bar(i32 %v7) #0
  %v9 = icmp eq i32 %v6, 10
  br i1 %v9, label %b13, label %b4

b10:                                              ; preds = %b1
  %v11 = add nsw i32 %a0, 65
  %v12 = tail call i32 @bar(i32 %v11) #0
  br label %b14

b13:                                              ; preds = %b4
  br label %b14

b14:                                              ; preds = %b13, %b10
  %v15 = phi i32 [ %v12, %b10 ], [ 0, %b13 ]
  ret i32 %v15
}

declare i32 @bar(i32) local_unnamed_addr #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b,-long-calls" }

!0 = !{!"branch_weights", i32 1, i32 2000}
