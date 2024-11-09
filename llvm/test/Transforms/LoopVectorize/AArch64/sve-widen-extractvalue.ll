; RUN: opt -S -passes=loop-vectorize < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @widen_extractvalue(ptr %dst, {i64, i64} %sv) #0 {
; CHECK-LABEL: @widen_extractvalue(
; CHECK:        [[EXTRACT0:%.*]] = extractvalue { i64, i64 } [[SV:%.*]], 0
; CHECK-NEXT:   [[EXTRACT1:%.*]] = extractvalue { i64, i64 } [[SV]], 1
; CHECK-NEXT:   [[DOTSPLATINSERT:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[EXTRACT0]], i64 0
; CHECK-NEXT:   [[DOTSPLAT:%.*]] = shufflevector <vscale x 2 x i64> [[DOTSPLATINSERT]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK-NEXT:   [[DOTSPLATINSERT1:%.*]] = insertelement <vscale x 2 x i64> poison, i64 [[EXTRACT1]], i64 0
; CHECK-NEXT:   [[DOTSPLAT2:%.*]] = shufflevector <vscale x 2 x i64> [[DOTSPLATINSERT1]], <vscale x 2 x i64> poison, <vscale x 2 x i32> zeroinitializer
; CHECK:        [[ADD:%.*]] = add <vscale x 2 x i64> [[DOTSPLAT]], [[DOTSPLAT2]]
; CHECK: vector.body:
entry:
  br label %loop.body

loop.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.body ]
  %a = extractvalue { i64, i64 } %sv, 0
  %b = extractvalue { i64, i64 } %sv, 1
  %addr = getelementptr i64, ptr %dst, i32 %iv
  %add = add i64 %a, %b
  store i64 %add, ptr %addr
  %iv.next = add nsw i32 %iv, 1
  %cond = icmp ne i32 %iv.next, 0
  br i1 %cond, label %loop.body, label %exit, !llvm.loop !0

exit:
  ret void
}

attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 2}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.interleave.count", i32 1}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}

