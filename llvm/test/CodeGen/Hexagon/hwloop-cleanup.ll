; RUN: llc -march=hexagon -no-phi-elim-live-out-early-exit < %s | FileCheck %s
; Check that we remove the compare and induction variable instructions
; after generating hardware loops.
; Bug 6685.

; CHECK-LABEL: f0:
; CHECK: loop0
; CHECK-NOT: r{{[0-9]+}} = add(r{{[0-9]+}},#-1)
; CHECK-NOT: cmp.eq
; CHECK: endloop0

define i32 @f0(i32* nocapture %a0, i32 %a1) #0 {
b0:
  %v0 = icmp sgt i32 %a1, 0
  br i1 %v0, label %b1, label %b4

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %v5, %b2 ], [ 0, %b1 ]
  %v2 = phi i32* [ %v8, %b2 ], [ %a0, %b1 ]
  %v3 = phi i32 [ %v6, %b2 ], [ 0, %b1 ]
  %v4 = load i32, i32* %v2, align 4
  %v5 = add nsw i32 %v4, %v1
  %v6 = add nsw i32 %v3, 1
  %v7 = icmp eq i32 %v6, %a1
  %v8 = getelementptr i32, i32* %v2, i32 1
  br i1 %v7, label %b3, label %b2

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b3, %b0
  %v9 = phi i32 [ 0, %b0 ], [ %v5, %b3 ]
  ret i32 %v9
}

; This test checks that that initial loop count value is removed.
; CHECK-LABEL: f1:
; CHECK-NOT: ={{.}}#40
; CHECK: loop0
; CHECK-NOT: r{{[0-9]+}} = add(r{{[0-9]+}},#-1)
; CHECK-NOT: cmp.eq
; CHECK: endloop0

define i32 @f1(i32* nocapture %a0) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v4, %b1 ]
  %v1 = phi i32* [ %a0, %b0 ], [ %v7, %b1 ]
  %v2 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v3 = load i32, i32* %v1, align 4
  %v4 = add nsw i32 %v3, %v0
  %v5 = add nsw i32 %v2, 1
  %v6 = icmp eq i32 %v5, 40
  %v7 = getelementptr i32, i32* %v1, i32 1
  br i1 %v6, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 %v4
}

; This test checks that we don't remove the induction variable since it's used.
; CHECK-LABEL: f2:
; CHECK: loop0
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}},#1)
; CHECK-NOT: cmp.eq
; CHECK: endloop0

define i32 @f2(i32* nocapture %a0) #1 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32* [ %a0, %b0 ], [ %v4, %b1 ]
  %v1 = phi i32 [ 0, %b0 ], [ %v2, %b1 ]
  store i32 %v1, i32* %v0, align 4
  %v2 = add nsw i32 %v1, 1
  %v3 = icmp eq i32 %v2, 40
  %v4 = getelementptr i32, i32* %v0, i32 1
  br i1 %v3, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 0
}

attributes #0 = { nounwind readonly "target-cpu"="hexagonv5" }
attributes #1 = { nounwind "target-cpu"="hexagonv5" }
