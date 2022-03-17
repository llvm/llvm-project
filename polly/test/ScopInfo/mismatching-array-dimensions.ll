; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s

; CHECK-NOT: AssumedContext

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define void @hoge([38 x [64 x float]]* %arg, [32 x [2 x float]]* %arg5, i32 %arg6) #0 {
bb:
  br i1 undef, label %bb7, label %bb25

bb7:                                              ; preds = %bb21, %bb
  %tmp8 = phi i64 [ %tmp22, %bb21 ], [ 0, %bb ]
  %tmp9 = icmp sgt i32 %arg6, 0
  br i1 %tmp9, label %bb10, label %bb21

bb10:                                             ; preds = %bb10, %bb7
  %tmp11 = getelementptr inbounds [32 x [2 x float]], [32 x [2 x float]]* %arg5, i64 %tmp8, i64 0
  %tmp = bitcast [2 x float]* %tmp11 to i32*
  %tmp12 = load i32, i32* %tmp, align 4, !tbaa !4
  %tmp13 = getelementptr inbounds [32 x [2 x float]], [32 x [2 x float]]* %arg5, i64 %tmp8, i64 0, i64 1
  %tmp14 = bitcast float* %tmp13 to i32*
  %tmp15 = load i32, i32* %tmp14, align 4, !tbaa !4
  %tmp16 = getelementptr inbounds [38 x [64 x float]], [38 x [64 x float]]* %arg, i64 1, i64 0, i64 %tmp8
  %tmp17 = bitcast float* %tmp16 to i32*
  store i32 %tmp15, i32* %tmp17, align 4, !tbaa !4
  %tmp18 = add nuw nsw i64 0, 1
  %tmp19 = trunc i64 %tmp18 to i32
  %tmp20 = icmp ne i32 %tmp19, %arg6
  br i1 %tmp20, label %bb10, label %bb21

bb21:                                             ; preds = %bb10, %bb7
  %tmp22 = add nsw i64 %tmp8, 1
  %tmp23 = trunc i64 %tmp22 to i32
  %tmp24 = icmp ne i32 %tmp23, 64
  br i1 %tmp24, label %bb7, label %bb25

bb25:                                             ; preds = %bb21, %bb
  ret void
}

attributes #0 = { nounwind ssp uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+ssse3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Dwarf Version", i32 2}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"PIC Level", i32 2}
!3 = !{!"clang version 3.8.0 (trunk 251760) (llvm/trunk 251765)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
