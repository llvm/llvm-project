; RUN: opt -instcombine -S < %s -mtriple=x86_64-apple-macosx10.9 | FileCheck %s --check-prefix=CHECK-FLOAT-IN-VEC
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-ios7.0 | FileCheck %s
; RUN: opt -instcombine -S < %s -mtriple=arm64-apple-ios7.0 | FileCheck %s
; RUN: opt -instcombine -S < %s -mtriple=x86_64-apple-macosx10.8 | FileCheck %s --check-prefix=CHECK-NO-SINCOS
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-ios6.0 | FileCheck %s --check-prefix=CHECK-NO-SINCOS
; RUN: opt -instcombine -S < %s -mtriple=x86_64-none-linux-gnu | FileCheck %s --check-prefix=CHECK-NO-SINCOS


attributes #0 = { readnone nounwind }

declare float @__sinpif(float %x) #0
declare float @__cospif(float %x) #0 

declare double @__sinpi(double %x) #0
declare double @__cospi(double %x) #0 

@var32 = global float 0.0
@var64 = global double 0.0

define float @test_instbased_f32() {
       %val = load float, float* @var32
       %sin = call float @__sinpif(float %val) #0
       %cos = call float @__cospif(float %val) #0
       %res = fadd float %sin, %cos
       ret float %res
; CHECK-FLOAT-IN-VEC: [[VAL:%[a-z0-9]+]] = load float, float* @var32
; CHECK-FLOAT-IN-VEC: [[SINCOS:%[a-z0-9]+]] = call <2 x float> @__sincospif_stret(float [[VAL]])
; CHECK-FLOAT-IN-VEC: extractelement <2 x float> [[SINCOS]], i32 0
; CHECK-FLOAT-IN-VEC: extractelement <2 x float> [[SINCOS]], i32 1

; CHECK: [[VAL:%[a-z0-9]+]] = load float, float* @var32
; CHECK: [[SINCOS:%[a-z0-9]+]] = call { float, float } @__sincospif_stret(float [[VAL]])
; CHECK: extractvalue { float, float } [[SINCOS]], 0
; CHECK: extractvalue { float, float } [[SINCOS]], 1

; CHECK-NO-SINCOS: call float @__sinpif
; CHECK-NO-SINCOS: call float @__cospif
}

define float @test_constant_f32() {
       %sin = call float @__sinpif(float 1.0) #0
       %cos = call float @__cospif(float 1.0) #0
       %res = fadd float %sin, %cos
       ret float %res
; CHECK-FLOAT-IN-VEC: [[SINCOS:%[a-z0-9]+]] = call <2 x float> @__sincospif_stret(float 1.000000e+00)
; CHECK-FLOAT-IN-VEC: extractelement <2 x float> [[SINCOS]], i32 0
; CHECK-FLOAT-IN-VEC: extractelement <2 x float> [[SINCOS]], i32 1

; CHECK: [[SINCOS:%[a-z0-9]+]] = call { float, float } @__sincospif_stret(float 1.000000e+00)
; CHECK: extractvalue { float, float } [[SINCOS]], 0
; CHECK: extractvalue { float, float } [[SINCOS]], 1

; CHECK-NO-SINCOS: call float @__sinpif
; CHECK-NO-SINCOS: call float @__cospif
}

define double @test_instbased_f64() {
       %val = load double, double* @var64
       %sin = call double @__sinpi(double %val) #0
       %cos = call double @__cospi(double %val) #0
       %res = fadd double %sin, %cos
       ret double %res
; CHECK-FLOAT-IN-VEC: [[VAL:%[a-z0-9]+]] = load double, double* @var64
; CHECK-FLOAT-IN-VEC: [[SINCOS:%[a-z0-9]+]] = call { double, double } @__sincospi_stret(double [[VAL]])
; CHECK-FLOAT-IN-VEC: extractvalue { double, double } [[SINCOS]], 0
; CHECK-FLOAT-IN-VEC: extractvalue { double, double } [[SINCOS]], 1

; CHECK: [[VAL:%[a-z0-9]+]] = load double, double* @var64
; CHECK: [[SINCOS:%[a-z0-9]+]] = call { double, double } @__sincospi_stret(double [[VAL]])
; CHECK: extractvalue { double, double } [[SINCOS]], 0
; CHECK: extractvalue { double, double } [[SINCOS]], 1

; CHECK-NO-SINCOS: call double @__sinpi
; CHECK-NO-SINCOS: call double @__cospi
}

define double @test_constant_f64() {
       %sin = call double @__sinpi(double 1.0) #0
       %cos = call double @__cospi(double 1.0) #0
       %res = fadd double %sin, %cos
       ret double %res
; CHECK-FLOAT-IN-VEC: [[SINCOS:%[a-z0-9]+]] = call { double, double } @__sincospi_stret(double 1.000000e+00)
; CHECK-FLOAT-IN-VEC: extractvalue { double, double } [[SINCOS]], 0
; CHECK-FLOAT-IN-VEC: extractvalue { double, double } [[SINCOS]], 1

; CHECK: [[SINCOS:%[a-z0-9]+]] = call { double, double } @__sincospi_stret(double 1.000000e+00)
; CHECK: extractvalue { double, double } [[SINCOS]], 0
; CHECK: extractvalue { double, double } [[SINCOS]], 1

; CHECK-NO-SINCOS: call double @__sinpi
; CHECK-NO-SINCOS: call double @__cospi
}
