; RUN: llc -march=hexagon -O3 < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: loop0
; a < b
define void @f0(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 32623, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ 32623, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 1
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f1:
; CHECK: loop0
; a < b
define void @f1(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 29554, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ 29554, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 2
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f2:
; CHECK: loop0
; a < b
define void @f2(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 15692, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ 15692, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 4
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f3:
; CHECK: loop0
; a < b
define void @f3(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 10449, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ 10449, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 8
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f4:
; CHECK: loop0
; a < b
define void @f4(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 32087, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ 32087, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 16
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f5:
; CHECK: loop0
; a < b
define void @f5(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, 3472
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 1
  %v8 = icmp ne i32 %v7, 3472
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f6:
; CHECK: loop0
; a < b
define void @f6(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, 8730
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 2
  %v8 = icmp ne i32 %v7, 8730
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f7:
; CHECK: loop0
; a < b
define void @f7(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, 1493
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 4
  %v8 = icmp ne i32 %v7, 1493
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f8:
; CHECK: loop0
; a < b
define void @f8(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, 1706
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 8
  %v8 = icmp ne i32 %v7, 1706
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f9:
; CHECK: loop0
; a < b
define void @f9(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, 1886
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 16
  %v8 = icmp ne i32 %v7, 1886
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f10:
; CHECK: loop0
; a < b
define void @f10(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 1
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f11:
; CHECK: loop0
; a < b
define void @f11(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 2
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f12:
; CHECK: loop0
; a < b
define void @f12(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 4
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f13
; CHECK: loop0
; a < b
define void @f13(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 8
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

; CHECK-LABEL: f14
; CHECK: loop0
; a < b
define void @f14(ptr nocapture %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = icmp slt i32 %a1, %a2
  br i1 %v0, label %b1, label %b3

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b2, %b1
  %v1 = phi i32 [ %a1, %b1 ], [ %v7, %b2 ]
  %v2 = getelementptr inbounds i8, ptr %a0, i32 %v1
  %v3 = load i8, ptr %v2, align 1
  %v4 = zext i8 %v3 to i32
  %v5 = add nsw i32 %v4, 1
  %v6 = trunc i32 %v5 to i8
  store i8 %v6, ptr %v2, align 1
  %v7 = add nsw i32 %v1, 16
  %v8 = icmp ne i32 %v7, %a2
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
