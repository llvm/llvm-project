; Broken by r326208
; XFAIL: *
; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; CHECK-NOT: add(r{{[0-9]+}},#2)
; CHECK-NOT: add(r{{[0-9]+}},#3)
; CHECK: memub(r{{[0-9]+}}+#2)
; CHECK: memub(r{{[0-9]+}}+#3)

@g0 = external global i32, align 4

define i32 @f0(i8* nocapture readonly %a0, i8* nocapture readonly %a1) {
b0:
  %v0 = getelementptr inbounds i8, i8* %a0, i32 2
  %v1 = getelementptr inbounds i8, i8* %a1, i32 3
  br label %b2

b1:                                               ; preds = %b3
  %v2 = getelementptr inbounds i8, i8* %a0, i32 %v7
  %v3 = add nuw nsw i32 %v7, 1
  %v4 = getelementptr inbounds i8, i8* %a1, i32 %v3
  %v5 = icmp eq i32 %v7, 3
  br i1 %v5, label %b4, label %b2

b2:                                               ; preds = %b1, %b0
  %v6 = phi i8* [ %v1, %b0 ], [ %v4, %b1 ]
  %v7 = phi i32 [ 3, %b0 ], [ %v3, %b1 ]
  %v8 = phi i8* [ %v0, %b0 ], [ %v2, %b1 ]
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v9 = load i8, i8* %v8, align 1
  %v10 = zext i8 %v9 to i32
  %v11 = load i8, i8* %v6, align 1
  %v12 = zext i8 %v11 to i32
  %v13 = tail call i32 bitcast (i32 (...)* @f1 to i32 (i32, i32)*)(i32 %v10, i32 %v12)
  %v14 = icmp eq i32 %v13, 0
  br i1 %v14, label %b1, label %b3

b4:                                               ; preds = %b1
  %v15 = tail call i32 @f2(i8* %a0, i8* %a1)
  %v16 = icmp sgt i32 %v15, 0
  br i1 %v16, label %b5, label %b6

b5:                                               ; preds = %b4
  store i32 10, i32* @g0, align 4
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v17 = phi i32 [ 1, %b5 ], [ 0, %b4 ]
  ret i32 %v17
}

declare i32 @f1(...)

declare i32 @f2(i8* nocapture, i8* nocapture)
