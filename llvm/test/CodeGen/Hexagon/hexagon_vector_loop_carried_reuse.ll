; RUN: opt < %s -hexagon-vlcr -adce -S | FileCheck %s

; CHECK: %.hexagon.vlcr = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B
; ModuleID = 'hexagon_vector_loop_carried_reuse.c'
source_filename = "hexagon_vector_loop_carried_reuse.c"
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

@W = external local_unnamed_addr global i32, align 4

; Function Attrs: nounwind
define void @foo(i8* noalias nocapture readonly %src, i8* noalias nocapture %dst, i32 %stride) local_unnamed_addr #0 {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %src, i32 %stride
  %mul = mul nsw i32 %stride, 2
  %add.ptr1 = getelementptr inbounds i8, i8* %src, i32 %mul
  %0 = load i32, i32* @W, align 4, !tbaa !1
  %cmp55 = icmp sgt i32 %0, 0
  br i1 %cmp55, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %1 = bitcast i8* %add.ptr1 to <32 x i32>*
  %2 = load <32 x i32>, <32 x i32>* %1, align 128, !tbaa !5
  %incdec.ptr4 = getelementptr inbounds i8, i8* %add.ptr1, i32 128
  %3 = bitcast i8* %incdec.ptr4 to <32 x i32>*
  %4 = bitcast i8* %add.ptr to <32 x i32>*
  %5 = load <32 x i32>, <32 x i32>* %4, align 128, !tbaa !5
  %incdec.ptr2 = getelementptr inbounds i8, i8* %add.ptr, i32 128
  %6 = bitcast i8* %incdec.ptr2 to <32 x i32>*
  %7 = bitcast i8* %src to <32 x i32>*
  %8 = load <32 x i32>, <32 x i32>* %7, align 128, !tbaa !5
  %incdec.ptr = getelementptr inbounds i8, i8* %src, i32 128
  %9 = bitcast i8* %incdec.ptr to <32 x i32>*
  %10 = bitcast i8* %dst to <32 x i32>*
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %out.063 = phi <32 x i32>* [ %10, %for.body.lr.ph ], [ %incdec.ptr18, %for.body ]
  %p2.062 = phi <32 x i32>* [ %3, %for.body.lr.ph ], [ %incdec.ptr10, %for.body ]
  %p1.061 = phi <32 x i32>* [ %6, %for.body.lr.ph ], [ %incdec.ptr8, %for.body ]
  %p0.060 = phi <32 x i32>* [ %9, %for.body.lr.ph ], [ %incdec.ptr6, %for.body ]
  %i.059 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %a.sroa.0.058 = phi <32 x i32> [ %8, %for.body.lr.ph ], [ %11, %for.body ]
  %b.sroa.0.057 = phi <32 x i32> [ %5, %for.body.lr.ph ], [ %12, %for.body ]
  %c.sroa.0.056 = phi <32 x i32> [ %2, %for.body.lr.ph ], [ %13, %for.body ]
  %incdec.ptr6 = getelementptr inbounds <32 x i32>, <32 x i32>* %p0.060, i32 1
  %11 = load <32 x i32>, <32 x i32>* %p0.060, align 128, !tbaa !5
  %incdec.ptr8 = getelementptr inbounds <32 x i32>, <32 x i32>* %p1.061, i32 1
  %12 = load <32 x i32>, <32 x i32>* %p1.061, align 128, !tbaa !5
  %incdec.ptr10 = getelementptr inbounds <32 x i32>, <32 x i32>* %p2.062, i32 1
  %13 = load <32 x i32>, <32 x i32>* %p2.062, align 128, !tbaa !5
  %14 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %a.sroa.0.058, <32 x i32> %b.sroa.0.057)
  %15 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %14, <32 x i32> %c.sroa.0.056)
  %16 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %11, <32 x i32> %12)
  %17 = tail call <32 x i32> @llvm.hexagon.V6.vmaxub.128B(<32 x i32> %16, <32 x i32> %13)
  %18 = tail call <32 x i32> @llvm.hexagon.V6.valignbi.128B(<32 x i32> %17, <32 x i32> %15, i32 1)
  %incdec.ptr18 = getelementptr inbounds <32 x i32>, <32 x i32>* %out.063, i32 1
  store <32 x i32> %18, <32 x i32>* %out.063, align 128, !tbaa !5
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
