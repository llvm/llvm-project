; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @cpp_new_arrays(
;CHECK: sext i32
;CHECK: load <4 x float>
;CHECK: fadd <4 x float>
;CHECK: ret i32
define i32 @cpp_new_arrays() uwtable ssp {
entry:
  %call = call noalias i8* @_Znwm(i64 4)
  %0 = bitcast i8* %call to float*
  store float 1.000000e+03, float* %0, align 4
  %call1 = call noalias i8* @_Znwm(i64 4)
  %1 = bitcast i8* %call1 to float*
  store float 1.000000e+03, float* %1, align 4
  %call3 = call noalias i8* @_Znwm(i64 4)
  %2 = bitcast i8* %call3 to float*
  store float 1.000000e+03, float* %2, align 4
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %idxprom = sext i32 %i.01 to i64
  %arrayidx = getelementptr inbounds float, float* %0, i64 %idxprom
  %3 = load float, float* %arrayidx, align 4
  %idxprom5 = sext i32 %i.01 to i64
  %arrayidx6 = getelementptr inbounds float, float* %1, i64 %idxprom5
  %4 = load float, float* %arrayidx6, align 4
  %add = fadd float %3, %4
  %idxprom7 = sext i32 %i.01 to i64
  %arrayidx8 = getelementptr inbounds float, float* %2, i64 %idxprom7
  store float %add, float* %arrayidx8, align 4
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 1000
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %5 = load float, float* %2, align 4
  %conv10 = fptosi float %5 to i32
  ret i32 %conv10
}

declare noalias i8* @_Znwm(i64)
