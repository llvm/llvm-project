; RUN: opt < %s -hexagon-vlcr | opt -passes=adce -S | FileCheck %s

; CHECK-NOT: %.hexagon.vlcr
; ModuleID = 'hexagon_vector_loop_carried_reuse.c'
source_filename = "hexagon_vector_loop_carried_reuse.c"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@W = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind
define void @foo(ptr noalias nocapture readonly %src, ptr noalias nocapture %dst, i32 %stride) local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, ptr %src, i32 %stride
  %mul = mul nsw i32 %stride, 2
  %add.ptr1 = getelementptr inbounds i8, ptr %src, i32 %mul
  %0 = load i32, ptr @W, align 4, !tbaa !1
  %cmp55 = icmp sgt i32 %0, 0
  br i1 %cmp55, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %1 = load <32 x i32>, ptr %add.ptr1, align 128, !tbaa !5
  %incdec.ptr4 = getelementptr inbounds i8, ptr %add.ptr1, i32 128
  %2 = load <32 x i32>, ptr %add.ptr, align 128, !tbaa !5
  %incdec.ptr2 = getelementptr inbounds i8, ptr %add.ptr, i32 128
  %3 = load <32 x i32>, ptr %src, align 128, !tbaa !5
  %incdec.ptr = getelementptr inbounds i8, ptr %src, i32 128
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %out.063 = phi ptr [ %dst, %for.body.lr.ph ], [ %incdec.ptr18, %for.body ]
  %p2.062 = phi ptr [ %incdec.ptr4, %for.body.lr.ph ], [ %incdec.ptr10, %for.body ]
  %p1.061 = phi ptr [ %incdec.ptr2, %for.body.lr.ph ], [ %incdec.ptr8, %for.body ]
  %p0.060 = phi ptr [ %incdec.ptr, %for.body.lr.ph ], [ %incdec.ptr6, %for.body ]
  %i.059 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %a.sroa.0.058 = phi <32 x i32> [ %3, %for.body.lr.ph ], [ %4, %for.body ]
  %b.sroa.0.057 = phi <32 x i32> [ %2, %for.body.lr.ph ], [ %5, %for.body ]
  %c.sroa.0.056 = phi <32 x i32> [ %1, %for.body.lr.ph ], [ %6, %for.body ]
  %incdec.ptr6 = getelementptr inbounds <32 x i32>, ptr %p0.060, i32 1
  %4 = load <32 x i32>, ptr %p0.060, align 128, !tbaa !5
  %incdec.ptr8 = getelementptr inbounds <32 x i32>, ptr %p1.061, i32 1
  %5 = load <32 x i32>, ptr %p1.061, align 128, !tbaa !5
  %incdec.ptr10 = getelementptr inbounds <32 x i32>, ptr %p2.062, i32 1
  %6 = load <32 x i32>, ptr %p2.062, align 128, !tbaa !5
  %7 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> %a.sroa.0.058, <32 x i32> %b.sroa.0.057, i32 4)
  %8 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %7, <32 x i32> %c.sroa.0.056)
  %9 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> %4, <32 x i32> %5, i32 5)
  %10 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %9, <32 x i32> %6)
  %11 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> %10, <32 x i32> %8, i32 1)
  %incdec.ptr18 = getelementptr inbounds <32 x i32>, ptr %out.063, i32 1
  store <32 x i32> %11, ptr %out.063, align 128, !tbaa !5
  %add = add nuw nsw i32 %i.059, 128
  %cmp = icmp slt i32 %add, %0
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32>, <32 x i32>) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32>, <32 x i32>, i32) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0}

!0 = !{!"QuIC LLVM Hexagon Clang version hexagon-clang-82-2622 (based on LLVM 5.0.0)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!3, !3, i64 0}
