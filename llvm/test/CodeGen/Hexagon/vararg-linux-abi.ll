; RUN: llc -march=hexagon -mtriple=hexagon-unknown-linux-musl < %s | FileCheck %s

; Check that we update the stack pointer before we do allocframe, so that
; the LR/FP are stored in the location required by the Linux ABI.
; CHECK: r29 = add(r29,#-24)
; CHECK: allocframe

target triple = "hexagon-unknown-linux"

%s.0 = type { i8*, i8*, i8* }

define dso_local i32 @f0(i32 %a0, ...) local_unnamed_addr #0 {
b0:
  %v0 = alloca [1 x %s.0], align 8
  %v1 = bitcast [1 x %s.0]* %v0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %v1) #2
  call void @llvm.va_start(i8* nonnull %v1)
  %v2 = getelementptr inbounds [1 x %s.0], [1 x %s.0]* %v0, i32 0, i32 0, i32 0
  %v3 = load i8*, i8** %v2, align 8
  %v4 = getelementptr inbounds [1 x %s.0], [1 x %s.0]* %v0, i32 0, i32 0, i32 1
  %v5 = load i8*, i8** %v4, align 4
  %v6 = getelementptr i8, i8* %v3, i32 4
  %v7 = icmp sgt i8* %v6, %v5
  br i1 %v7, label %b1, label %b2

b1:                                               ; preds = %b0
  %v8 = getelementptr inbounds [1 x %s.0], [1 x %s.0]* %v0, i32 0, i32 0, i32 2
  %v9 = load i8*, i8** %v8, align 8
  %v10 = getelementptr i8, i8* %v9, i32 4
  store i8* %v10, i8** %v8, align 8
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v11 = phi i8* [ %v10, %b1 ], [ %v6, %b0 ]
  %v12 = phi i8* [ %v9, %b1 ], [ %v3, %b0 ]
  %v13 = bitcast i8* %v12 to i32*
  store i8* %v11, i8** %v2, align 8
  %v14 = load i32, i32* %v13, align 4
  %v15 = icmp eq i32 %v14, 0
  br i1 %v15, label %b7, label %b3

b3:                                               ; preds = %b2
  %v16 = getelementptr inbounds [1 x %s.0], [1 x %s.0]* %v0, i32 0, i32 0, i32 2
  br label %b4

b4:                                               ; preds = %b6, %b3
  %v17 = phi i32 [ %v14, %b3 ], [ %v28, %b6 ]
  %v18 = phi i32 [ %a0, %b3 ], [ %v20, %b6 ]
  %v19 = phi i8* [ %v11, %b3 ], [ %v25, %b6 ]
  %v20 = add nsw i32 %v17, %v18
  %v21 = getelementptr i8, i8* %v19, i32 4
  %v22 = icmp sgt i8* %v21, %v5
  br i1 %v22, label %b5, label %b6

b5:                                               ; preds = %b4
  %v23 = load i8*, i8** %v16, align 8
  %v24 = getelementptr i8, i8* %v23, i32 4
  store i8* %v24, i8** %v16, align 8
  br label %b6

b6:                                               ; preds = %b5, %b4
  %v25 = phi i8* [ %v24, %b5 ], [ %v21, %b4 ]
  %v26 = phi i8* [ %v23, %b5 ], [ %v19, %b4 ]
  %v27 = bitcast i8* %v26 to i32*
  store i8* %v25, i8** %v2, align 8
  %v28 = load i32, i32* %v27, align 4
  %v29 = icmp eq i32 %v28, 0
  br i1 %v29, label %b7, label %b4

b7:                                               ; preds = %b6, %b2
  %v30 = phi i32 [ %a0, %b2 ], [ %v20, %b6 ]
  call void @llvm.va_end(i8* nonnull %v1)
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %v1) #2
  ret i32 %v30
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #2

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { argmemonly nounwind "frame-pointer"="all" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
