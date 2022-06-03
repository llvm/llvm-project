; RUN: opt -aa-pipeline=basic-aa -passes='loop(loop-interchange)'       -S %s | FileCheck %s --check-prefixes INTC
; RUN: opt -aa-pipeline=basic-aa -passes='loop-mssa(lnicm),loop(loop-interchange)' -S %s | FileCheck %s --check-prefixes LNICM,CHECK
; RUN: opt -aa-pipeline=basic-aa -passes='loop-mssa(licm),loop(loop-interchange)'  -S %s | FileCheck %s --check-prefixes LICM,CHECK

; This test represents the following function:
; void test(int x[10][10], int y[10], int *z) {
;   for (int k = 0; k < 10; k++) {
;     int tmp = *z;
;     for (int i = 0; i < 10; i++)
;       x[i][k] += y[k] + tmp;
;   }
; }
; We only want to hoist the load of z out of the loop nest.
; LICM hoists the load of y[k] out of the i-loop, but LNICM doesn't do so
; to keep perfect loop nest. This enables optimizations that require
; perfect loop nest (e.g. loop-interchange) to perform.


define dso_local void @test([10 x i32]* noalias %x, i32* noalias readonly %y, i32* readonly %z) {
; CHECK-LABEL: @test(
; CHECK-NEXT: entry:
; CHECK-NEXT:   [[Z:%.*]] = load i32, i32* %z, align 4
; CHECK-NEXT:   br label [[FOR_BODY3_PREHEADER:%.*]]
; LNICM:      for.body.preheader:
; LICM-NOT:   for.body.preheader:
; INTC-NOT:   for.body.preheader:
; LNICM-NEXT:   br label [[FOR_BODY:%.*]]
; CHECK:      for.body:
; LNICM-NEXT:   [[K:%.*]] = phi i32 [ [[INC10:%.*]], [[FOR_END:%.*]] ], [ 0, [[FOR_BODY_PREHEADER:%.*]] ]
; LNICM-NEXT:   br label [[FOR_BODY3_SPLIT1:%.*]]
; LICM:         [[TMP:%.*]] = load i32, i32* [[ARRAYIDX:%.*]], align 4
; LNICM:      for.body3.preheader:
; LICM-NOT:   for.body3.preheader:
; INTC-NOT:   for.body3.preheader:
; LNICM-NEXT:   br label [[FOR_BODY3:%.*]]
; CHECK:      for.body3:
; LNICM-NEXT:   [[I:%.*]] = phi i32 [ [[TMP3:%.*]], [[FOR_BODY3_SPLIT:%.*]] ], [ 0, [[FOR_BODY3_PREHEADER:%.*]] ]
; LNICM-NEXT:   br label [[FOR_BODY_PREHEADER:%.*]]
; LNICM:      for.body3.split1:
; LNICM-NEXT:   [[IDXPROM:%.*]] = sext i32 [[K:%.*]] to i64
; LNICM-NEXT:   [[ARRAYIDX:%.*]] = getelementptr inbounds i32, i32* %y, i64 [[IDXPROM:%.*]]
; LNICM-NEXT:   [[TMP:%.*]] = load i32, i32* [[ARRAYIDX:%.*]], align 4
; LNICM-NEXT:   [[ADD:%.*]] = add nsw i32 [[TMP:%.*]], [[Z:%.*]]
; LNICM-NEXT:   [[IDXPROM4:%.*]] = sext i32 [[I:%.*]] to i64
; LNICM-NEXT:   [[ARRAYIDX5:%.*]] = getelementptr inbounds [10 x i32], [10 x i32]* %x, i64 [[IDXPROM4:%.*]]
; LNICM-NEXT:   [[IDXPROM6:%.*]] = sext i32 [[K:%.*]] to i64
; LNICM-NEXT:   [[ARRAYIDX7:%.*]] = getelementptr inbounds [10 x i32], [10 x i32]* [[ARRAYIDX5:%.*]], i64 0, i64 [[IDXPROM6:%.*]]
; LNICM-NEXT:   [[TMP2:%.*]] = load i32, i32* [[ARRAYIDX7:%.*]], align 4
; LNICM-NEXT:   [[ADD8:%.*]] = add nsw i32 [[TMP2:%.*]], [[ADD:%.*]]
; LNICM-NEXT:   store i32 [[ADD8:%.*]], i32* [[ARRAYIDX7:%.*]], align 4
; LNICM-NEXT:   [[INC:%.*]] = add nsw i32 [[I:%.*]], 1
; LNICM-NEXT:   [[CMP2:%.*]] = icmp slt i32 [[INC:%.*]], 10
; LNICM-NEXT:   br label [[FOR_END:%.*]]
; LNICM:      for.body3.split:
; LICM-NOT:   for.body3.split:
; INTC-NOT:   for.body3.split:
; LNICM-NEXT:   [[TMP3:%.*]] = add nsw i32 [[I:%.*]], 1
; LNICM-NEXT:   [[TMP4:%.*]] = icmp slt i32 [[TMP3:%.*]], 10
; LNICM-NEXT:   br i1 [[TMP4:%.*]], label [[FOR_BODY3:%.*]], label [[FOR_END11:%.*]], !llvm.loop !0
; LNICM:      for.end:
; LNICM-NEXT:   [[INC10:%.*]] = add nsw i32 [[K:%.*]], 1
; LNICM-NEXT:   [[CMP:%.*]] = icmp slt i32 [[INC10:%.*]], 10
; LNICM-NEXT:   br i1 [[CMP:%.*]], label [[FOR_BODY:%.*]], label [[FOR_BODY3_SPLIT:%.*]], !llvm.loop !2
; LNICM:      for.end11:
; LNICM-NEXT:   ret void

entry:
  br label %for.body

for.body:
  %k.02 = phi i32 [ 0, %entry ], [ %inc10, %for.end ]
  %0 = load i32, i32* %z, align 4
  br label %for.body3

for.body3:
  %i.01 = phi i32 [ 0, %for.body ], [ %inc, %for.body3 ]
  %idxprom = sext i32 %k.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %y, i64 %idxprom
  %1 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %1, %0
  %idxprom4 = sext i32 %i.01 to i64
  %arrayidx5 = getelementptr inbounds [10 x i32], [10 x i32]* %x, i64 %idxprom4
  %idxprom6 = sext i32 %k.02 to i64
  %arrayidx7 = getelementptr inbounds [10 x i32], [10 x i32]* %arrayidx5, i64 0, i64 %idxprom6
  %2 = load i32, i32* %arrayidx7, align 4
  %add8 = add nsw i32 %2, %add
  store i32 %add8, i32* %arrayidx7, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp2 = icmp slt i32 %inc, 10
  br i1 %cmp2, label %for.body3, label %for.end, !llvm.loop !0

for.end:
  %inc10 = add nsw i32 %k.02, 1
  %cmp = icmp slt i32 %inc10, 10
  br i1 %cmp, label %for.body, label %for.end11, !llvm.loop !2

for.end11:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
!2 = distinct !{!2, !1}
