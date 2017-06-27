; RUN: opt -tbaa -basicaa -gvn -S < %s | FileCheck %s
; PR9971

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.X = type { i32, float }
%union.vector_t = type { [2 x i64] }

; Don't delete the load after the loop, because it loads values stored
; inside the loop.

; CHECK: define void @vrlh(

; CHECK: for.end:
; CHECK:   %arrayidx31 = getelementptr inbounds %union.vector_t, %union.vector_t* %t, i64 0, i32 0, i64 1
; CHECK:   %tmp32 = load i64, i64* %arrayidx31, align 8, !tbaa [[TAG:!.*]]

define void @vrlh(%union.vector_t* %va, %union.vector_t* %vb, %union.vector_t* %vd) nounwind {
entry:
  %t = alloca %union.vector_t, align 8
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sub = sub nsw i32 7, %i.01
  %idxprom = sext i32 %sub to i64
  %half = bitcast %union.vector_t* %vb to [8 x i16]*
  %arrayidx = getelementptr inbounds [8 x i16], [8 x i16]* %half, i64 0, i64 %idxprom
  %tmp4 = load i16, i16* %arrayidx, align 2, !tbaa !10
  %conv = zext i16 %tmp4 to i32
  %and = and i32 %conv, 15
  %sub6 = sub nsw i32 7, %i.01
  %idxprom7 = sext i32 %sub6 to i64
  %half9 = bitcast %union.vector_t* %va to [8 x i16]*
  %arrayidx10 = getelementptr inbounds [8 x i16], [8 x i16]* %half9, i64 0, i64 %idxprom7
  %tmp11 = load i16, i16* %arrayidx10, align 2, !tbaa !10
  %conv12 = zext i16 %tmp11 to i32
  %shl = shl i32 %conv12, %and
  %sub15 = sub nsw i32 7, %i.01
  %idxprom16 = sext i32 %sub15 to i64
  %half18 = bitcast %union.vector_t* %va to [8 x i16]*
  %arrayidx19 = getelementptr inbounds [8 x i16], [8 x i16]* %half18, i64 0, i64 %idxprom16
  %tmp20 = load i16, i16* %arrayidx19, align 2, !tbaa !10
  %conv21 = zext i16 %tmp20 to i32
  %sub23 = sub nsw i32 16, %and
  %shr = lshr i32 %conv21, %sub23
  %or = or i32 %shl, %shr
  %conv24 = trunc i32 %or to i16
  %sub26 = sub nsw i32 7, %i.01
  %idxprom27 = sext i32 %sub26 to i64
  %half28 = bitcast %union.vector_t* %t to [8 x i16]*
  %arrayidx29 = getelementptr inbounds [8 x i16], [8 x i16]* %half28, i64 0, i64 %idxprom27
  store i16 %conv24, i16* %arrayidx29, align 2, !tbaa !10
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 8
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %arrayidx31 = getelementptr inbounds %union.vector_t, %union.vector_t* %t, i64 0, i32 0, i64 1
  %tmp32 = load i64, i64* %arrayidx31, align 8, !tbaa !10
  %arrayidx35 = getelementptr inbounds %union.vector_t, %union.vector_t* %vd, i64 0, i32 0, i64 1
  store i64 %tmp32, i64* %arrayidx35, align 8, !tbaa !10
  %arrayidx37 = getelementptr inbounds %union.vector_t, %union.vector_t* %t, i64 0, i32 0, i64 0
  %tmp38 = load i64, i64* %arrayidx37, align 8, !tbaa !10
  %arrayidx41 = getelementptr inbounds %union.vector_t, %union.vector_t* %vd, i64 0, i32 0, i64 0
  store i64 %tmp38, i64* %arrayidx41, align 8, !tbaa !10
  ret void
}

; Do delete the load after the loop.

; CHECK: define i32 @test0(

; CHECK:   ret i32 0

define i32 @test0(%struct.X* %a) nounwind {
entry:
  %i = getelementptr inbounds %struct.X, %struct.X* %a, i64 0, i32 0
  store i32 0, i32* %i, align 4, !tbaa !4
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i2.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %f = getelementptr inbounds %struct.X, %struct.X* %a, i64 %i2.01, i32 1
  %tmp6 = load float, float* %f, align 4, !tbaa !5
  %mul = fmul float %tmp6, 0x40019999A0000000
  store float %mul, float* %f, align 4, !tbaa !5
  %inc = add nsw i64 %i2.01, 1
  %cmp = icmp slt i64 %inc, 10000
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %i9 = getelementptr inbounds %struct.X, %struct.X* %a, i64 0, i32 0
  %tmp10 = load i32, i32* %i9, align 4, !tbaa !4
  ret i32 %tmp10
}

; Do delete the load after the loop.

; CHECK: define float @test1(

; CHECK:   ret float 0x3FD3333340000000

define float @test1(%struct.X* %a) nounwind {
entry:
  %f = getelementptr inbounds %struct.X, %struct.X* %a, i64 0, i32 1
  store float 0x3FD3333340000000, float* %f, align 4, !tbaa !5
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %i5 = getelementptr inbounds %struct.X, %struct.X* %a, i64 %i.01, i32 0
  %tmp6 = load i32, i32* %i5, align 4, !tbaa !4
  %mul = mul nsw i32 %tmp6, 3
  store i32 %mul, i32* %i5, align 4, !tbaa !4
  %inc = add nsw i64 %i.01, 1
  %cmp = icmp slt i64 %inc, 10000
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %f9 = getelementptr inbounds %struct.X, %struct.X* %a, i64 0, i32 1
  %tmp10 = load float, float* %f9, align 4, !tbaa !5
  ret float %tmp10
}

; CHECK: [[TAG]] = !{[[TYPE_LL:!.*]], [[TYPE_LL]], i64 0}
; CHECK: [[TYPE_LL]] = !{!"omnipotent char", {{!.*}}}
!0 = !{!6, !6, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!7, !7, i64 0}
!4 = !{!8, !8, i64 0}
!5 = !{!9, !9, i64 0}
!6 = !{!"short", !1}
!7 = !{!"long long", !1}
!8 = !{!"int", !1}
!9 = !{!"float", !1}
!10 = !{!1, !1, i64 0}
