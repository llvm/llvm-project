; RUN: llc < %s -march=xcore | FileCheck %s
declare double @llvm.cos.f64(double)
declare double @llvm.exp.f64(double)
declare double @llvm.exp2.f64(double)
declare double @llvm.log.f64(double)
declare double @llvm.log10.f64(double)
declare double @llvm.log2.f64(double)
declare double @llvm.pow.f64(double, double)
declare double @llvm.powi.f64.i32(double, i32)
declare double @llvm.sin.f64(double)
declare double @llvm.sqrt.f64(double)

define double @cos(double %F) {
; CHECK-LABEL: cos:
; CHECK: bl cos
        %result = call double @llvm.cos.f64(double %F)
	ret double %result
}

declare float @llvm.cos.f32(float)

; CHECK-LABEL: cosf:
; CHECK: bl cosf
define float @cosf(float %F) {
        %result = call float @llvm.cos.f32(float %F)
	ret float %result
}

define double @exp(double %F) {
; CHECK-LABEL: exp:
; CHECK: bl exp
        %result = call double @llvm.exp.f64(double %F)
	ret double %result
}

declare float @llvm.exp.f32(float)

define float @expf(float %F) {
; CHECK-LABEL: expf:
; CHECK: bl expf
        %result = call float @llvm.exp.f32(float %F)
	ret float %result
}

define double @exp2(double %F) {
; CHECK-LABEL: exp2:
; CHECK: bl exp2
        %result = call double @llvm.exp2.f64(double %F)
	ret double %result
}

declare float @llvm.exp2.f32(float)

define float @exp2f(float %F) {
; CHECK-LABEL: exp2f:
; CHECK: bl exp2f
        %result = call float @llvm.exp2.f32(float %F)
	ret float %result
}

define double @log(double %F) {
; CHECK-LABEL: log:
; CHECK: bl log
        %result = call double @llvm.log.f64(double %F)
	ret double %result
}

declare float @llvm.log.f32(float)

define float @logf(float %F) {
; CHECK-LABEL: logf:
; CHECK: bl logf
        %result = call float @llvm.log.f32(float %F)
	ret float %result
}

define double @log10(double %F) {
; CHECK-LABEL: log10:
; CHECK: bl log10
        %result = call double @llvm.log10.f64(double %F)
	ret double %result
}

declare float @llvm.log10.f32(float)

define float @log10f(float %F) {
; CHECK-LABEL: log10f:
; CHECK: bl log10f
        %result = call float @llvm.log10.f32(float %F)
	ret float %result
}

define double @log2(double %F) {
; CHECK-LABEL: log2:
; CHECK: bl log2
        %result = call double @llvm.log2.f64(double %F)
	ret double %result
}

declare float @llvm.log2.f32(float)

define float @log2f(float %F) {
; CHECK-LABEL: log2f:
; CHECK: bl log2f
        %result = call float @llvm.log2.f32(float %F)
	ret float %result
}

define double @pow(double %F, double %power) {
; CHECK-LABEL: pow:
; CHECK: bl pow
        %result = call double @llvm.pow.f64(double %F, double %power)
	ret double %result
}

declare float @llvm.pow.f32(float, float)

define float @powf(float %F, float %power) {
; CHECK-LABEL: powf:
; CHECK: bl powf
        %result = call float @llvm.pow.f32(float %F, float %power)
	ret float %result
}

define double @powi(double %F, i32 %power) {
; CHECK-LABEL: powi:
; CHECK: bl __powidf2
        %result = call double @llvm.powi.f64.i32(double %F, i32 %power)
	ret double %result
}

declare float @llvm.powi.f32.i32(float, i32)

define float @powif(float %F, i32 %power) {
; CHECK-LABEL: powif:
; CHECK: bl __powisf2
        %result = call float @llvm.powi.f32.i32(float %F, i32 %power)
	ret float %result
}

define double @sin(double %F) {
; CHECK-LABEL: sin:
; CHECK: bl sin
        %result = call double @llvm.sin.f64(double %F)
	ret double %result
}

declare float @llvm.sin.f32(float)

define float @sinf(float %F) {
; CHECK-LABEL: sinf:
; CHECK: bl sinf
        %result = call float @llvm.sin.f32(float %F)
	ret float %result
}

define double @sqrt(double %F) {
; CHECK-LABEL: sqrt:
; CHECK: bl sqrt
        %result = call double @llvm.sqrt.f64(double %F)
	ret double %result
}

declare float @llvm.sqrt.f32(float)

define float @sqrtf(float %F) {
; CHECK-LABEL: sqrtf:
; CHECK: bl sqrtf
        %result = call float @llvm.sqrt.f32(float %F)
	ret float %result
}
