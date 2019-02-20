; RUN: llc -march=hexagon -O2 < %s | FileCheck %s
; This test checks if redundant conditional branches are removed.

; CHECK: memub
; CHECK: memub
; CHECK: memub
; CHECK-NOT: if{{.*}}jump .LBB
; CHECK: cmp.eq

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
declare void @f0() #0

; Function Attrs: nounwind
define void @f1(i8* %a0, i32 %a1, i8* %a2, i32* %a3) #0 {
b0:
  br i1 undef, label %b8, label %b1

b1:                                               ; preds = %b0
  tail call void @f0() #0
  br i1 false, label %b8, label %b2

b2:                                               ; preds = %b1
  %v0 = getelementptr inbounds i8, i8* %a0, i32 undef
  %v1 = sub i32 0, %a1
  %v2 = icmp eq i32 undef, undef
  br label %b3

b3:                                               ; preds = %b6, %b2
  %v3 = phi i8* [ %a2, %b2 ], [ %v17, %b6 ]
  %v4 = phi i8* [ %v0, %b2 ], [ null, %b6 ]
  %v5 = phi i32 [ 1, %b2 ], [ 0, %b6 ]
  br i1 %v2, label %b4, label %b5

b4:                                               ; preds = %b3
  %v6 = load i8, i8* %v3, align 1
  br label %b6

b5:                                               ; preds = %b3
  %v7 = load i8, i8* %v4, align 1
  %v8 = zext i8 %v7 to i32
  %v9 = getelementptr inbounds i8, i8* %v4, i32 %v1
  %v10 = load i8, i8* %v9, align 1
  %v11 = zext i8 %v10 to i32
  %v12 = sub nsw i32 %v8, %v11
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v13 = phi i8 [ 0, %b5 ], [ %v6, %b4 ]
  %v14 = phi i32 [ %v12, %b5 ], [ 0, %b4 ]
  %v15 = zext i8 %v13 to i32
  %v16 = mul nsw i32 %v14, %v14
  %v17 = getelementptr inbounds i8, i8* %v3, i32 1
  %v18 = sub nsw i32 0, %v15
  %v19 = mul nsw i32 %v18, %v18
  %v20 = add nuw i32 %v16, 0
  %v21 = add i32 %v20, 0
  %v22 = add i32 %v21, 0
  %v23 = lshr i32 %v22, 1
  %v24 = add nuw nsw i32 %v23, %v19
  %v25 = add nsw i32 %v24, 0
  store i32 %v25, i32* %a3, align 4
  %v26 = icmp eq i32 %v5, undef
  br i1 %v26, label %b7, label %b3

b7:                                               ; preds = %b6
  ret void

b8:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }
