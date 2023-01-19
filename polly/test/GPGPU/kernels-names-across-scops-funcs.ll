; RUN: opt %loadPolly -polly-process-unprofitable -polly-codegen-ppcg \
; RUN: -polly-acc-dump-kernel-ir -disable-output < %s | \
; RUN: FileCheck -check-prefix=KERNEL %s

; REQUIRES: pollyacc

; KERNEL: define ptx_kernel void @FUNC_foo_SCOP_0_KERNEL_0(ptr addrspace(1) %MemRef_arg1, i32 %arg) #0 {
; KERNEL: define ptx_kernel void @FUNC_foo_SCOP_1_KERNEL_0(ptr addrspace(1) %MemRef_arg1, i32 %arg) #0 {
; KERNEL: define ptx_kernel void @FUNC_foo2_SCOP_0_KERNEL_0(ptr addrspace(1) %MemRef_arg1, i32 %arg) #0 {

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @foo(i32 %arg, ptr %arg1) #0 {
bb:
  br label %bb2

bb2:                                              ; preds = %bb
  %tmp = icmp sgt i32 %arg, 0
  br i1 %tmp, label %bb3, label %bb13

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb4, %bb3
  %tmp5 = phi i64 [ 0, %bb3 ], [ %tmp9, %bb4 ]
  %tmp6 = getelementptr inbounds i32, ptr %arg1, i64 %tmp5
  %tmp7 = load i32, ptr %tmp6, align 4, !tbaa !2
  %tmp8 = add nsw i32 %tmp7, 1
  store i32 %tmp8, ptr %tmp6, align 4, !tbaa !2
  %tmp9 = add nuw nsw i64 %tmp5, 1
  %tmp10 = zext i32 %arg to i64
  %tmp11 = icmp ne i64 %tmp9, %tmp10
  br i1 %tmp11, label %bb4, label %bb12

bb12:                                             ; preds = %bb4
  br label %bb13

bb13:                                             ; preds = %bb12, %bb2
  %tmp14 = tail call i64 @clock() #3
  %tmp15 = icmp eq i64 %tmp14, 0
  br i1 %tmp15, label %bb16, label %bb29

bb16:                                             ; preds = %bb13
  %tmp17 = icmp sgt i32 %arg, 0
  br i1 %tmp17, label %bb18, label %bb28

bb18:                                             ; preds = %bb16
  br label %bb19

bb19:                                             ; preds = %bb19, %bb18
  %tmp20 = phi i64 [ 0, %bb18 ], [ %tmp24, %bb19 ]
  %tmp21 = getelementptr inbounds i32, ptr %arg1, i64 %tmp20
  %tmp22 = load i32, ptr %tmp21, align 4, !tbaa !2
  %tmp23 = add nsw i32 %tmp22, 1
  store i32 %tmp23, ptr %tmp21, align 4, !tbaa !2
  %tmp24 = add nuw nsw i64 %tmp20, 1
  %tmp25 = zext i32 %arg to i64
  %tmp26 = icmp ne i64 %tmp24, %tmp25
  br i1 %tmp26, label %bb19, label %bb27

bb27:                                             ; preds = %bb19
  br label %bb28

bb28:                                             ; preds = %bb27, %bb16
  br label %bb29

bb29:                                             ; preds = %bb28, %bb13
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind
declare i64 @clock() #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind uwtable
define void @foo2(i32 %arg, ptr %arg1) #0 {
bb:
  br label %bb2

bb2:                                              ; preds = %bb
  %tmp = icmp sgt i32 %arg, 0
  br i1 %tmp, label %bb3, label %bb13

bb3:                                              ; preds = %bb2
  br label %bb4

bb4:                                              ; preds = %bb4, %bb3
  %tmp5 = phi i64 [ 0, %bb3 ], [ %tmp9, %bb4 ]
  %tmp6 = getelementptr inbounds i32, ptr %arg1, i64 %tmp5
  %tmp7 = load i32, ptr %tmp6, align 4, !tbaa !2
  %tmp8 = add nsw i32 %tmp7, 1
  store i32 %tmp8, ptr %tmp6, align 4, !tbaa !2
  %tmp9 = add nuw nsw i64 %tmp5, 1
  %tmp10 = zext i32 %arg to i64
  %tmp11 = icmp ne i64 %tmp9, %tmp10
  br i1 %tmp11, label %bb4, label %bb12

bb12:                                             ; preds = %bb4
  br label %bb13

bb13:                                             ; preds = %bb12, %bb2
  ret void
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
