; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK: cmp.eq
; CHECK: cmp.eq
; CHECK: cmp.eq
; CHECK: cmp.eq

%s.0 = type { ptr, i32, ptr }

@g0 = external global ptr, align 4
@g1 = private global [4 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8

declare void @f0(ptr)

define i32 @f1() #0 {
b0:
  %v0 = load i64, ptr @g1, align 8
  %v1 = add i64 %v0, 1
  store i64 %v1, ptr @g1, align 8
  br label %b1

b1:                                               ; preds = %b6, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v27, %b6 ]
  %v3 = load i64, ptr getelementptr inbounds ([4 x i64], ptr @g1, i32 0, i32 1), align 8
  %v4 = add i64 %v3, 1
  store i64 %v4, ptr getelementptr inbounds ([4 x i64], ptr @g1, i32 0, i32 1), align 8
  %v5 = load ptr, ptr @g0, align 4
  %v6 = getelementptr inbounds ptr, ptr %v5, i32 %v2
  %v7 = load ptr, ptr %v6, align 4
  %v8 = icmp eq ptr %v7, null
  br i1 %v8, label %b6, label %b2

b2:                                               ; preds = %b1
  %v9 = load i64, ptr getelementptr inbounds ([4 x i64], ptr @g1, i32 0, i32 2), align 8
  %v10 = add i64 %v9, 1
  store i64 %v10, ptr getelementptr inbounds ([4 x i64], ptr @g1, i32 0, i32 2), align 8
  %v12 = getelementptr inbounds %s.0, ptr %v7, i32 0, i32 2
  %v13 = load ptr, ptr %v12, align 4
  %v14 = icmp eq ptr %v13, null
  %v15 = getelementptr inbounds %s.0, ptr %v7, i32 0, i32 2
  br i1 %v14, label %b5, label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v16 = phi ptr [ %v25, %b4 ], [ %v15, %b3 ]
  %v17 = phi ptr [ %v20, %b4 ], [ %v7, %b3 ]
  %v18 = load i64, ptr getelementptr inbounds ([4 x i64], ptr @g1, i32 0, i32 3), align 8
  %v19 = add i64 %v18, 1
  store i64 %v19, ptr getelementptr inbounds ([4 x i64], ptr @g1, i32 0, i32 3), align 8
  %v20 = load ptr, ptr %v16, align 4
  tail call void @f0(ptr %v17)
  %v22 = getelementptr inbounds %s.0, ptr %v20, i32 0, i32 2
  %v23 = load ptr, ptr %v22, align 4
  %v24 = icmp eq ptr %v23, null
  %v25 = getelementptr inbounds %s.0, ptr %v20, i32 0, i32 2
  br i1 %v24, label %b5, label %b4

b5:                                               ; preds = %b4, %b2
  %v26 = phi ptr [ %v7, %b2 ], [ %v20, %b4 ]
  tail call void @f0(ptr %v26)
  br label %b6

b6:                                               ; preds = %b5, %b1
  %v27 = add nuw nsw i32 %v2, 1
  %v28 = icmp eq i32 %v27, 3001
  br i1 %v28, label %b7, label %b1

b7:                                               ; preds = %b6
  %v29 = load ptr, ptr @g0, align 4
  tail call void @f0(ptr %v29)
  ret i32 undef
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }

