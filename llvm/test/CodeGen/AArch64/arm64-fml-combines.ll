; RUN: llc < %s -O3 -mtriple=arm64-apple-ios -enable-unsafe-fp-math -mattr=+fullfp16 | FileCheck %s
; RUN: llc < %s -O3 -mtriple=arm64-apple-ios -fp-contract=fast -mattr=+fullfp16 | FileCheck %s

define void @foo_2d(double* %src) {
entry:
  %arrayidx1 = getelementptr inbounds double, double* %src, i64 5
  %arrayidx2 = getelementptr inbounds double, double* %src, i64 11
  %tmp = bitcast double* %arrayidx1 to <2 x double>*
  br label %for.body

; CHECK-LABEL: %for.body
; CHECK: fmls.2d {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; CHECK: fmls.2d {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}[0]
; CHECK: fmls.d {{d[0-9]+}}, {{d[0-9]+}}, {{v[0-9]+}}[0]
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = sub nuw nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds double, double* %src, i64 %indvars.iv.next
  %tmp1 = load double, double* %arrayidx3, align 8
  %add = fadd fast double %tmp1, %tmp1
  %mul = fmul fast double %add, %add
  %e1 = insertelement <2 x double> undef, double %add, i32 0
  %e2 = insertelement <2 x double> %e1, double %add, i32 1
  %sub2 = fsub fast <2 x double> %e2, <double 3.000000e+00, double -3.000000e+00>
  %e3 = insertelement <2 x double> undef, double %mul, i32 0
  %e4 = insertelement <2 x double> %e3, double %mul, i32 1
  %mul2 = fmul fast <2 x double> %sub2,<double 3.000000e+00, double -3.000000e+00>
  %e5 = insertelement <2 x double> undef, double %add, i32 0
  %e6 = insertelement <2 x double> %e5, double %add, i32 1
  %sub3 = fsub fast  <2 x double>  <double 3.000000e+00, double -3.000000e+00>, %mul2
  %mulx = fmul fast <2 x double> %sub2, %e2
  %subx = fsub fast  <2 x double> %e4, %mulx
  %e7 = insertelement <2 x double> undef, double %mul, i32 0
  %e8 = insertelement <2 x double> %e7, double %mul, i32 1
  %e9 = fmul fast <2 x double>  %subx, %sub3
  store <2 x double> %e9, <2 x double>* %tmp, align 8
  %e10 = extractelement <2 x double> %sub3, i32 0
  %mul3 = fmul fast double %mul, %e10
  %sub4 = fsub fast double %mul, %mul3
  store double %sub4, double* %arrayidx2, align 8
  %exitcond = icmp eq i64 %indvars.iv.next, 25
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
define void @foo_2s(float* %src) {
entry:
  %arrayidx1 = getelementptr inbounds float, float* %src, i64 5
  %arrayidx2 = getelementptr inbounds float, float* %src, i64 11
  %tmp = bitcast float* %arrayidx1 to <2 x float>*
  br label %for.body

; CHECK-LABEL: %for.body
; CHECK: fmls.2s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; CHECK: fmls.2s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}[0]
; CHECK: fmls.s {{s[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}[0]
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds float, float* %src, i64 %indvars.iv.next
  %tmp1 = load float, float* %arrayidx3, align 8
  %add = fadd fast float %tmp1, %tmp1
  %mul = fmul fast float %add, %add
  %e1 = insertelement <2 x float> undef, float %add, i32 0
  %e2 = insertelement <2 x float> %e1, float %add, i32 1
  %add2 = fsub fast <2 x float> %e2, <float 3.000000e+00, float -3.000000e+00>
  %e3 = insertelement <2 x float> undef, float %mul, i32 0
  %e4 = insertelement <2 x float> %e3, float %mul, i32 1
  %mul2 = fmul fast <2 x float> %add2,<float 3.000000e+00, float -3.000000e+00>
  %e5 = insertelement <2 x float> undef, float %add, i32 0
  %e6 = insertelement <2 x float> %e5, float %add, i32 1
  %add3 = fsub fast  <2 x float>  <float 3.000000e+00, float -3.000000e+00>, %mul2
  %mulx = fmul fast <2 x float> %add2, %e2
  %addx = fsub fast  <2 x float> %e4, %mulx
  %e7 = insertelement <2 x float> undef, float %mul, i32 0
  %e8 = insertelement <2 x float> %e7, float %mul, i32 1
  %e9 = fmul fast <2 x float>  %addx, %add3
  store <2 x float> %e9, <2 x float>* %tmp, align 8
  %e10 = extractelement <2 x float> %add3, i32 0
  %mul3 = fmul fast float %mul, %e10
  %add4 = fsub fast float %mul, %mul3
  store float %add4, float* %arrayidx2, align 8
  %exitcond = icmp eq i64 %indvars.iv.next, 25
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
define void @foo_4s(float* %src) {
entry:
  %arrayidx1 = getelementptr inbounds float, float* %src, i64 5
  %arrayidx2 = getelementptr inbounds float, float* %src, i64 11
  %tmp = bitcast float* %arrayidx1 to <4 x float>*
  br label %for.body

; CHECK-LABEL: %for.body
; CHECK: fmls.4s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; CHECK: fmls.4s {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}[0]
for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx3 = getelementptr inbounds float, float* %src, i64 %indvars.iv.next
  %tmp1 = load float, float* %arrayidx3, align 8
  %add = fadd fast float %tmp1, %tmp1
  %mul = fmul fast float %add, %add
  %e1 = insertelement <4 x float> undef, float %add, i32 0
  %e2 = insertelement <4 x float> %e1, float %add, i32 1
  %add2 = fadd fast <4 x float> %e2, <float 3.000000e+00, float -3.000000e+00, float 5.000000e+00, float 7.000000e+00>
  %e3 = insertelement <4 x float> undef, float %mul, i32 0
  %e4 = insertelement <4 x float> %e3, float %mul, i32 1
  %mul2 = fmul fast <4 x float> %add2,<float 3.000000e+00, float -3.000000e+00, float 5.000000e+00, float 7.000000e+00>
  %e5 = insertelement <4 x float> undef, float %add, i32 0
  %e6 = insertelement <4 x float> %e5, float %add, i32 1
  %add3 = fsub fast  <4 x float> <float 3.000000e+00, float -3.000000e+00, float 5.000000e+00, float 7.000000e+00> , %mul2
  %mulx = fmul fast <4 x float> %add2, %e2
  %addx = fsub fast  <4 x float> %e4, %mulx
  %e7 = insertelement <4 x float> undef, float %mul, i32 0
  %e8 = insertelement <4 x float> %e7, float %mul, i32 1
  %e9 = fmul fast <4 x float>  %addx, %add3
  store <4 x float> %e9, <4 x float>* %tmp, align 8
  %e10 = extractelement <4 x float> %add3, i32 0
  %mul3 = fmul fast float %mul, %e10
  store float %mul3, float* %arrayidx2, align 8
  %exitcond = icmp eq i64 %indvars.iv.next, 25
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

; CHECK-LABEL: test0:
; CHECK: fnmadd h0, h0, h1, h2
define half @test0(half %a, half %b, half %c) {
entry:
  %0 = fmul half %a, %b
  %mul = fsub half -0.000000e+00, %0
  %sub1 = fsub half %mul, %c
  ret half %sub1
}

; CHECK-LABEL: test1:
; CHECK: fnmadd s0, s0, s1, s2
define float @test1(float %a, float %b, float %c) {
entry:
  %0 = fmul float %a, %b
  %mul = fsub float -0.000000e+00, %0
  %sub1 = fsub float %mul, %c
  ret float %sub1
}

; CHECK-LABEL: test2:
; CHECK: fnmadd d0, d0, d1, d2
define double @test2(double %a, double %b, double %c) {
entry:
  %0 = fmul double %a, %b
  %mul = fsub double -0.000000e+00, %0
  %sub1 = fsub double %mul, %c
  ret double %sub1
}

; CHECK-LABEL: test3:
; CHECK: fnmadd h0, h0, h1, h2
define half @test3(half %0, half %1, half %2) {
  %4 = fneg fast half %0
  %5 = fmul fast half %4, %1
  %6 = fsub fast half %5, %2
  ret half %6
}

; CHECK-LABEL: test4:
; CHECK: fnmadd s0, s0, s1, s2
define float @test4(float %0, float %1, float %2) {
  %4 = fneg fast float %0
  %5 = fmul fast float %4, %1
  %6 = fsub fast float %5, %2
  ret float %6
}

; CHECK-LABEL: test5:
; CHECK: fnmadd d0, d0, d1, d2
define double @test5(double %0, double %1, double %2) {
  %4 = fneg fast double %0
  %5 = fmul fast double %4, %1
  %6 = fsub fast double %5, %2
  ret double %6
}
