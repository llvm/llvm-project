; RUN: llc -march=hexagon -mtriple=hexagon-unknown-linux-musl < %s | FileCheck %s

; Check that we update the stack pointer before we do allocframe, so that
; the LR/FP are stored in the location required by the Linux ABI.
; CHECK: r29 = add(r29,#-24)
; CHECK: allocframe

target triple = "hexagon-unknown-linux"

%s.0 = type { ptr, ptr, ptr }

define dso_local i32 @f0(i32 %a0, ...) local_unnamed_addr #0 {
b0:
  %v0 = alloca [1 x %s.0], align 8
  call void @llvm.lifetime.start.p0(i64 12, ptr nonnull %v0) #2
  call void @llvm.va_start(ptr nonnull %v0)
  %v3 = load ptr, ptr %v0, align 8
  %v4 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 1
  %v5 = load ptr, ptr %v4, align 4
  %v6 = getelementptr i8, ptr %v3, i32 4
  %v7 = icmp sgt ptr %v6, %v5
  br i1 %v7, label %b1, label %b2

b1:                                               ; preds = %b0
  %v8 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 2
  %v9 = load ptr, ptr %v8, align 8
  %v10 = getelementptr i8, ptr %v9, i32 4
  store ptr %v10, ptr %v8, align 8
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v11 = phi ptr [ %v10, %b1 ], [ %v6, %b0 ]
  %v12 = phi ptr [ %v9, %b1 ], [ %v3, %b0 ]
  store ptr %v11, ptr %v0, align 8
  %v14 = load i32, ptr %v12, align 4
  %v15 = icmp eq i32 %v14, 0
  br i1 %v15, label %b7, label %b3

b3:                                               ; preds = %b2
  %v16 = getelementptr inbounds [1 x %s.0], ptr %v0, i32 0, i32 0, i32 2
  br label %b4

b4:                                               ; preds = %b6, %b3
  %v17 = phi i32 [ %v14, %b3 ], [ %v28, %b6 ]
  %v18 = phi i32 [ %a0, %b3 ], [ %v20, %b6 ]
  %v19 = phi ptr [ %v11, %b3 ], [ %v25, %b6 ]
  %v20 = add nsw i32 %v17, %v18
  %v21 = getelementptr i8, ptr %v19, i32 4
  %v22 = icmp sgt ptr %v21, %v5
  br i1 %v22, label %b5, label %b6

b5:                                               ; preds = %b4
  %v23 = load ptr, ptr %v16, align 8
  %v24 = getelementptr i8, ptr %v23, i32 4
  store ptr %v24, ptr %v16, align 8
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v25 = phi ptr [ %v24, %b5 ], [ %v21, %b4 ]
  %v26 = phi ptr [ %v23, %b5 ], [ %v19, %b4 ]
  store ptr %v25, ptr %v0, align 8
  %v28 = load i32, ptr %v26, align 4
  %v29 = icmp eq i32 %v28, 0
  br i1 %v29, label %b7, label %b4

b7:                                               ; preds = %b6, %b2
  %v30 = phi i32 [ %a0, %b2 ], [ %v20, %b6 ]
  call void @llvm.va_end(ptr nonnull %v0)
  call void @llvm.lifetime.end.p0(i64 12, ptr nonnull %v0) #2
  ret i32 %v30
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind
declare void @llvm.va_start(ptr) #2

; Function Attrs: nounwind
declare void @llvm.va_end(ptr) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

attributes #0 = { argmemonly nounwind "frame-pointer"="all" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
