; RUN: opt < %s -scalarrepl -S | FileCheck %s
; RUN: opt < %s -scalarrepl-ssa -S | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10.0.0"

%struct.Vector4 = type { float, float, float, float }
@f.vector = internal constant %struct.Vector4 { float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 }, align 16

; CHECK: define void @f
; CHECK-NOT: alloca
; CHECK: phi <4 x float>

define void @f() nounwind ssp {
entry:
  %i = alloca i32, align 4
  %vector = alloca %struct.Vector4, align 16
  %agg.tmp = alloca %struct.Vector4, align 16
  %tmp = bitcast %struct.Vector4* %vector to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* bitcast (%struct.Vector4* @f.vector to i8*), i32 16, i32 16, i1 false)
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %storemerge = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  store i32 %storemerge, i32* %i, align 4
  %cmp = icmp slt i32 %storemerge, 1000000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp2 = bitcast %struct.Vector4* %agg.tmp to i8*
  %tmp3 = bitcast %struct.Vector4* %vector to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp2, i8* %tmp3, i32 16, i32 16, i1 false)
  %0 = bitcast %struct.Vector4* %agg.tmp to [2 x i64]*
  %1 = load [2 x i64]* %0, align 16
  %tmp2.i = extractvalue [2 x i64] %1, 0
  %tmp3.i = zext i64 %tmp2.i to i128
  %tmp10.i = bitcast i128 %tmp3.i to <4 x float>
  %sub.i.i = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %tmp10.i
  %2 = bitcast %struct.Vector4* %vector to <4 x float>*
  store <4 x float> %sub.i.i, <4 x float>* %2, align 16
  %tmp4 = load i32* %i, align 4
  %inc = add nsw i32 %tmp4, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %x = getelementptr inbounds %struct.Vector4* %vector, i32 0, i32 0
  %tmp5 = load float* %x, align 16
  %conv = fpext float %tmp5 to double
  %call = call i32 (...)* @printf(double %conv) nounwind
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare i32 @printf(...)
