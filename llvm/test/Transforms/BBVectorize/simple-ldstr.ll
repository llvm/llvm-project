target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-aligned-only -instcombine -gvn -S | FileCheck %s -check-prefix=CHECK-AO

; Simple 3-pair chain with loads and stores
define void @test1(double* %a, double* %b, double* %c) nounwind uwtable readonly {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
; CHECK: @test1
; CHECK: %i0.v.i0 = bitcast double* %a to <2 x double>*
; CHECK: %i1.v.i0 = bitcast double* %b to <2 x double>*
; CHECK: %i0 = load <2 x double>* %i0.v.i0, align 8
; CHECK: %i1 = load <2 x double>* %i1.v.i0, align 8
; CHECK: %mul = fmul <2 x double> %i0, %i1
; CHECK: %0 = bitcast double* %c to <2 x double>*
; CHECK: store <2 x double> %mul, <2 x double>* %0, align 8
; CHECK: ret void
; CHECK-AO: @test1
; CHECK-AO-NOT: <2 x double>
}

; Simple chain with extending loads and stores
define void @test2(float* %a, float* %b, double* %c) nounwind uwtable readonly {
entry:
  %i0f = load float* %a, align 4
  %i0 = fpext float %i0f to double
  %i1f = load float* %b, align 4
  %i1 = fpext float %i1f to double
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds float* %a, i64 1
  %i3f = load float* %arrayidx3, align 4
  %i3 = fpext float %i3f to double
  %arrayidx4 = getelementptr inbounds float* %b, i64 1
  %i4f = load float* %arrayidx4, align 4
  %i4 = fpext float %i4f to double
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
; CHECK: @test2
; CHECK: %i0f.v.i0 = bitcast float* %a to <2 x float>*
; CHECK: %i1f.v.i0 = bitcast float* %b to <2 x float>*
; CHECK: %i0f = load <2 x float>* %i0f.v.i0, align 4
; CHECK: %i0 = fpext <2 x float> %i0f to <2 x double>
; CHECK: %i1f = load <2 x float>* %i1f.v.i0, align 4
; CHECK: %i1 = fpext <2 x float> %i1f to <2 x double>
; CHECK: %mul = fmul <2 x double> %i0, %i1
; CHECK: %0 = bitcast double* %c to <2 x double>*
; CHECK: store <2 x double> %mul, <2 x double>* %0, align 8
; CHECK: ret void
; CHECK-AO: @test2
; CHECK-AO-NOT: <2 x double>
}

; Simple chain with loads and truncating stores
define void @test3(double* %a, double* %b, float* %c) nounwind uwtable readonly {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %mulf = fptrunc double %mul to float
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %mul5f = fptrunc double %mul5 to float
  store float %mulf, float* %c, align 8
  %arrayidx5 = getelementptr inbounds float* %c, i64 1
  store float %mul5f, float* %arrayidx5, align 4
  ret void
; CHECK: @test3
; CHECK: %i0.v.i0 = bitcast double* %a to <2 x double>*
; CHECK: %i1.v.i0 = bitcast double* %b to <2 x double>*
; CHECK: %i0 = load <2 x double>* %i0.v.i0, align 8
; CHECK: %i1 = load <2 x double>* %i1.v.i0, align 8
; CHECK: %mul = fmul <2 x double> %i0, %i1
; CHECK: %mulf = fptrunc <2 x double> %mul to <2 x float>
; CHECK: %0 = bitcast float* %c to <2 x float>*
; CHECK: store <2 x float> %mulf, <2 x float>* %0, align 8
; CHECK: ret void
; CHECK-AO: @test3
; CHECK-AO: %i0 = load double* %a, align 8
; CHECK-AO: %i1 = load double* %b, align 8
; CHECK-AO: %arrayidx3 = getelementptr inbounds double* %a, i64 1
; CHECK-AO: %i3 = load double* %arrayidx3, align 8
; CHECK-AO: %arrayidx4 = getelementptr inbounds double* %b, i64 1
; CHECK-AO: %i4 = load double* %arrayidx4, align 8
; CHECK-AO: %mul.v.i1.1 = insertelement <2 x double> undef, double %i1, i32 0
; CHECK-AO: %mul.v.i1.2 = insertelement <2 x double> %mul.v.i1.1, double %i4, i32 1
; CHECK-AO: %mul.v.i0.1 = insertelement <2 x double> undef, double %i0, i32 0
; CHECK-AO: %mul.v.i0.2 = insertelement <2 x double> %mul.v.i0.1, double %i3, i32 1
; CHECK-AO: %mul = fmul <2 x double> %mul.v.i0.2, %mul.v.i1.2
; CHECK-AO: %mulf = fptrunc <2 x double> %mul to <2 x float>
; CHECK-AO: %0 = bitcast float* %c to <2 x float>*
; CHECK-AO: store <2 x float> %mulf, <2 x float>* %0, align 8
; CHECK-AO: ret void
}

; Simple 3-pair chain with loads and stores (unreachable)
define void @test4(i1 %bool, double* %a, double* %b, double* %c) nounwind uwtable readonly {
entry:
  br i1 %bool, label %if.then1, label %if.end

if.then1:
  unreachable
  br label %if.then

if.then:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  br label %if.end

if.end:
  ret void
; CHECK: @test4
; CHECK-NOT: <2 x double>
; CHECK-AO: @test4
; CHECK-AO-NOT: <2 x double>
}

; Simple 3-pair chain with loads and stores
define void @test5(double* %a, double* %b, double* %c) nounwind uwtable readonly {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  store double %mul, double* %c, align 4
  ret void
; CHECK: @test5
; CHECK: %i0.v.i0 = bitcast double* %a to <2 x double>*
; CHECK: %i1.v.i0 = bitcast double* %b to <2 x double>*
; CHECK: %i0 = load <2 x double>* %i0.v.i0, align 8
; CHECK: %i1 = load <2 x double>* %i1.v.i0, align 8
; CHECK: %mul = fmul <2 x double> %i0, %i1
; CHECK: %0 = bitcast double* %c to <2 x double>*
; CHECK: store <2 x double> %mul, <2 x double>* %0, align 4
; CHECK: ret void
; CHECK-AO: @test5
; CHECK-AO-NOT: <2 x double>
}

