; RUN: opt < %s -mcpu=pwr7 -mattr=+vsx -loop-vectorize -instcombine -S | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.GlobalData = type { [32000 x float], [3 x i32], [4 x i8], [32000 x float], [5 x i32], [12 x i8], [32000 x float], [7 x i32], [4 x i8], [32000 x float], [11 x i32], [4 x i8], [32000 x float], [13 x i32], [12 x i8], [256 x [256 x float]], [17 x i32], [12 x i8], [256 x [256 x float]], [19 x i32], [4 x i8], [256 x [256 x float]], [23 x i32], [4 x i8], [256 x [256 x float]] }

@global_data = external global %struct.GlobalData, align 16
@ntimes = external hidden unnamed_addr global i32, align 4

define signext i32 @s173() #0 {
entry:
  %0 = load i32, i32* @ntimes, align 4
  %cmp21 = icmp sgt i32 %0, 0
  br i1 %cmp21, label %for.cond1.preheader, label %for.end12

for.cond1.preheader:                              ; preds = %for.end, %entry
  %nl.022 = phi i32 [ %inc11, %for.end ], [ 0, %entry ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx = getelementptr inbounds %struct.GlobalData, %struct.GlobalData* @global_data, i64 0, i32 0, i64 %indvars.iv
  %1 = load float, float* %arrayidx, align 4
  %arrayidx5 = getelementptr inbounds %struct.GlobalData, %struct.GlobalData* @global_data, i64 0, i32 3, i64 %indvars.iv
  %2 = load float, float* %arrayidx5, align 4
  %add = fadd float %1, %2
  %3 = add nsw i64 %indvars.iv, 16000
  %arrayidx8 = getelementptr inbounds %struct.GlobalData, %struct.GlobalData* @global_data, i64 0, i32 0, i64 %3
  store float %add, float* %arrayidx8, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 16000
  br i1 %exitcond, label %for.end, label %for.body3

for.end:                                          ; preds = %for.body3
  %inc11 = add nsw i32 %nl.022, 1
  %4 = load i32, i32* @ntimes, align 4
  %mul = mul nsw i32 %4, 10
  %cmp = icmp slt i32 %inc11, %mul
  br i1 %cmp, label %for.cond1.preheader, label %for.end12

for.end12:                                        ; preds = %for.end, %entry
  ret i32 0

; CHECK-LABEL: @s173
; CHECK: load <4 x float>, <4 x float>*
; CHECK: add nsw i64 %index, 16000
; CHECK: ret i32 0
}

attributes #0 = { nounwind }

