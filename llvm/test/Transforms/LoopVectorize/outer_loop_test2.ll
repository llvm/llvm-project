; int A[1024], B[1024];
;
; void foo(int iCount, int c, int jCount)
; {
;
;   int i, j;
;
; #pragma clang loop vectorize(enable) vectorize_width(4)
;   for (i = 0; i < iCount; i++) {
;     A[i] = c;
;     for (j = 0; j < jCount; j++) {
;       A[i] += B[j] + i;
;     }
;   }
; }
; RUN: opt -S -passes=loop-vectorize -enable-vplan-native-path < %s | FileCheck %s
; CHECK: %[[ZeroTripChk:.*]] = icmp sgt i32 %jCount, 0
; CHECK-LABEL: vector.ph:
; CHECK: %[[CVal0:.*]] = insertelement <4 x i32> poison, i32 %c, i64 0
; CHECK-NEXT: %[[CSplat:.*]] = shufflevector <4 x i32> %[[CVal0]], <4 x i32> poison, <4 x i32> zeroinitializer

; CHECK-LABEL: vector.body:
; CHECK: %[[Ind:.*]] = phi i64 [ 0, %vector.ph ], [ %[[IndNext:.*]], %[[ForInc:.*]] ]
; CHECK: %[[VecInd:.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ %[[VecIndNext:.*]], %[[ForInc]] ]
; CHECK: %[[AAddr:.*]] = getelementptr inbounds [1024 x i32], ptr @A, i64 0, <4 x i64> %[[VecInd]]
; CHECK: call void @llvm.masked.scatter.v4i32.v4p0(<4 x i32> %[[CSplat]], <4 x ptr> align 4 %[[AAddr]], <4 x i1> splat (i1 true))
; CHECK: br i1 %[[ZeroTripChk]], label %[[InnerForPh:.*]], label %[[OuterInc:.*]]

; CHECK: [[InnerForPh]]:
; CHECK: %[[WideAVal:.*]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> align 4 %[[AAddr]], <4 x i1> splat (i1 true), <4 x i32> poison)
; CHECK: %[[VecIndTr:.*]] = trunc <4 x i64> %[[VecInd]] to <4 x i32>
; CHECK: br label %[[InnerForBody:.*]]

; CHECK: [[InnerForBody]]:
; CHECK: %[[InnerInd:.*]] = phi <4 x i64> [ zeroinitializer, %[[InnerForPh]] ], [ %[[InnerIndNext:.*]], %[[InnerForBody]] ]
; CHECK: %[[AccumPhi:.*]] = phi <4 x i32> [ %[[WideAVal]], %[[InnerForPh]] ], [ %[[AccumPhiNext:.*]], %[[InnerForBody]] ]
; CHECK: %[[BAddr:.*]] = getelementptr inbounds [1024 x i32], ptr @B, i64 0, <4 x i64> %[[InnerInd]]
; CHECK: %[[WideBVal:.*]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> align 4 %[[BAddr]], <4 x i1> splat (i1 true), <4 x i32> poison)
; CHECK: %[[Add1:.*]] = add nsw <4 x i32> %[[WideBVal]], %[[VecIndTr]]
; CHECK: %[[AccumPhiNext]] = add nsw <4 x i32> %[[Add1]], %[[AccumPhi]]
; CHECK: %[[InnerIndNext]] = add nuw nsw <4 x i64> %[[InnerInd]], splat (i64 1)
; CHECK: %[[InnerVecCond:.*]] = icmp eq <4 x i64> %[[InnerIndNext]], {{.*}}
; CHECK: %[[InnerCond:.+]] = extractelement <4 x i1> %[[InnerVecCond]], i32 0
; CHECK: br i1 %[[InnerCond]], label %[[InnerCrit:.*]], label %[[InnerForBody]]

; CHECK: [[InnerCrit]]:
; CHECK: %[[StorePhi:.*]] = phi <4 x i32> [ %[[AccumPhiNext]], %[[InnerForBody]] ]
; CHECK: call void @llvm.masked.scatter.v4i32.v4p0(<4 x i32> %[[StorePhi]], <4 x ptr> align 4 %[[AAddr]], <4 x i1> splat (i1 true))
; CHECK:  br label %[[ForInc]]

; CHECK: [[ForInc]]:
; CHECK: %[[IndNext]] = add nuw i64 %[[Ind]], 4
; CHECK: %[[VecIndNext]] = add <4 x i64> %[[VecInd]], splat (i64 4)
; CHECK: %[[Cmp:.*]] = icmp eq i64 %[[IndNext]], {{.*}}
; CHECK: br i1 %[[Cmp]], label %middle.block, label %vector.body

@A = common global [1024 x i32] zeroinitializer, align 16
@B = common global [1024 x i32] zeroinitializer, align 16

; Function Attrs: norecurse nounwind uwtable
define void @foo(i32 %iCount, i32 %c, i32 %jCount) {
entry:
  %cmp22 = icmp sgt i32 %iCount, 0
  br i1 %cmp22, label %for.body.lr.ph, label %for.end11

for.body.lr.ph:                                   ; preds = %entry
  %cmp220 = icmp sgt i32 %jCount, 0
  %wide.trip.count = zext i32 %jCount to i64
  %wide.trip.count27 = zext i32 %iCount to i64
  br label %for.body

for.body:                                         ; preds = %for.inc9, %for.body.lr.ph
  %indvars.iv25 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next26, %for.inc9 ]
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv25
  store i32 %c, ptr %arrayidx, align 4
  br i1 %cmp220, label %for.body3.lr.ph, label %for.inc9

for.body3.lr.ph:                                  ; preds = %for.body
  %arrayidx.promoted = load i32, ptr %arrayidx, align 4
  %0 = trunc i64 %indvars.iv25 to i32
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body3.lr.ph ], [ %indvars.iv.next, %for.body3 ]
  %1 = phi i32 [ %arrayidx.promoted, %for.body3.lr.ph ], [ %add8, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [1024 x i32], ptr @B, i64 0, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx5, align 4
  %add = add nsw i32 %2, %0
  %add8 = add nsw i32 %add, %1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond1.for.inc9_crit_edge, label %for.body3

for.cond1.for.inc9_crit_edge:                     ; preds = %for.body3
  store i32 %add8, ptr %arrayidx, align 4
  br label %for.inc9

for.inc9:                                         ; preds = %for.cond1.for.inc9_crit_edge, %for.body
  %indvars.iv.next26 = add nuw nsw i64 %indvars.iv25, 1
  %exitcond28 = icmp eq i64 %indvars.iv.next26, %wide.trip.count27
  br i1 %exitcond28, label %for.end11, label %for.body, !llvm.loop !1

for.end11:                                        ; preds = %for.inc9, %entry
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
