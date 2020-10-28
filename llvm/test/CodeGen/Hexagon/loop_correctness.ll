; RUN: llc -march=hexagon -O3 -hexagon-instsimplify=0 < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: loop0(.LBB{{[0-9]+}}_{{[0-9]+}},#3)
; CHECK: endloop0
define void @f0(i8* nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v1, %b2 ]
  %v1 = add nsw i32 %v0, 1
  %v2 = icmp slt i32 %v1, 3
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret void
}

; CHECK-LABEL: f1:
; CHECK: loop0(.LBB{{[0-9]+}}_{{[0-9]+}},#2)
; CHECK: endloop0
define void @f1(i8* nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v1, %b2 ]
  %v1 = add nsw i32 %v0, 2
  %v2 = icmp slt i32 %v1, 3
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret void
}

; CHECK-LABEL: f2:
; CHECK: loop0(.LBB{{[0-9]+}}_{{[0-9]+}},#1)
; CHECK: endloop0
define void @f2(i8* nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v1, %b2 ]
  %v1 = add nsw i32 %v0, 3
  %v2 = icmp slt i32 %v1, 3
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret void
}

; CHECK-LABEL: f3:
; CHECK: loop0(.LBB{{[0-9]+}}_{{[0-9]+}},#4)
; CHECK: endloop0
define void @f3(i8* nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v1, %b2 ]
  %v1 = add nsw i32 %v0, 1
  %v2 = icmp sle i32 %v1, 3
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret void
}

; CHECK-LABEL: f4:
; CHECK: loop0(.LBB{{[0-9]+}}_{{[0-9]+}},#2)
; CHECK: endloop0
define void @f4(i8* nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v1, %b2 ]
  %v1 = add nsw i32 %v0, 2
  %v2 = icmp sle i32 %v1, 3
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret void
}

; CHECK-LABEL: f5:
; CHECK: loop0(.LBB{{[0-9]+}}_{{[0-9]+}},#2)
; CHECK: endloop0
define void @f5(i8* nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  br label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v0 = phi i32 [ 0, %b1 ], [ %v1, %b2 ]
  %v1 = add nsw i32 %v0, 3
  %v2 = icmp sle i32 %v1, 3
  br i1 %v2, label %b2, label %b3

b3:                                               ; preds = %b2
  ret void
}

attributes #0 = { nounwind }
