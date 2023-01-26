; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; FP elimination enabled.
;
; CHECK-LABEL: danny:
; CHECK: r29 = add(r29,#-[[SIZE:[0-9]+]])
; CHECK: r29 = add(r29,#[[SIZE]])
define i32 @danny(i32 %a0, i32 %a1) local_unnamed_addr #0 {
b2:
  %v3 = alloca [32 x i32], align 8
  call void @llvm.lifetime.start.p0(i64 128, ptr nonnull %v3) #3
  br label %b5

b5:                                               ; preds = %b5, %b2
  %v6 = phi i32 [ 0, %b2 ], [ %v8, %b5 ]
  %v7 = getelementptr inbounds [32 x i32], ptr %v3, i32 0, i32 %v6
  store i32 %v6, ptr %v7, align 4
  %v8 = add nuw nsw i32 %v6, 1
  %v9 = icmp eq i32 %v8, 32
  br i1 %v9, label %b10, label %b5

b10:                                              ; preds = %b5
  %v11 = getelementptr inbounds [32 x i32], ptr %v3, i32 0, i32 %a0
  store i32 %a1, ptr %v11, align 4
  br label %b12

b12:                                              ; preds = %b12, %b10
  %v13 = phi i32 [ 0, %b10 ], [ %v18, %b12 ]
  %v14 = phi i32 [ 0, %b10 ], [ %v17, %b12 ]
  %v15 = getelementptr inbounds [32 x i32], ptr %v3, i32 0, i32 %v13
  %v16 = load i32, ptr %v15, align 4
  %v17 = add nsw i32 %v16, %v14
  %v18 = add nuw nsw i32 %v13, 1
  %v19 = icmp eq i32 %v18, 32
  br i1 %v19, label %b20, label %b12

b20:                                              ; preds = %b12
  call void @llvm.lifetime.end.p0(i64 128, ptr nonnull %v3) #3
  ret i32 %v17
}

; FP elimination disabled.
;
; CHECK-LABEL: sammy:
; CHECK: allocframe
; CHECK: dealloc_return
define i32 @sammy(i32 %a0, i32 %a1) local_unnamed_addr #1 {
b2:
  %v3 = alloca [32 x i32], align 8
  call void @llvm.lifetime.start.p0(i64 128, ptr nonnull %v3) #3
  br label %b5

b5:                                               ; preds = %b5, %b2
  %v6 = phi i32 [ 0, %b2 ], [ %v8, %b5 ]
  %v7 = getelementptr inbounds [32 x i32], ptr %v3, i32 0, i32 %v6
  store i32 %v6, ptr %v7, align 4
  %v8 = add nuw nsw i32 %v6, 1
  %v9 = icmp eq i32 %v8, 32
  br i1 %v9, label %b10, label %b5

b10:                                              ; preds = %b5
  %v11 = getelementptr inbounds [32 x i32], ptr %v3, i32 0, i32 %a0
  store i32 %a1, ptr %v11, align 4
  br label %b12

b12:                                              ; preds = %b12, %b10
  %v13 = phi i32 [ 0, %b10 ], [ %v18, %b12 ]
  %v14 = phi i32 [ 0, %b10 ], [ %v17, %b12 ]
  %v15 = getelementptr inbounds [32 x i32], ptr %v3, i32 0, i32 %v13
  %v16 = load i32, ptr %v15, align 4
  %v17 = add nsw i32 %v16, %v14
  %v18 = add nuw nsw i32 %v13, 1
  %v19 = icmp eq i32 %v18, 32
  br i1 %v19, label %b20, label %b12

b20:                                              ; preds = %b12
  call void @llvm.lifetime.end.p0(i64 128, ptr nonnull %v3) #3
  ret i32 %v17
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #2
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #2

attributes #0 = { nounwind readnone "frame-pointer"="none" "target-cpu"="hexagonv60" }
attributes #1 = { nounwind readnone "frame-pointer"="all" "target-cpu"="hexagonv60" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind }
