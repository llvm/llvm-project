; RUN: llc < %s | FileCheck %s --check-prefixes=CHECK,CHECK-NOF16
; RUN: llc < %s -mcpu=sm_80 | FileCheck %s --check-prefixes=CHECK,CHECK-F16
; RUN: llc < %s -mcpu=sm_80 --nvptx-no-f16-math | FileCheck %s --check-prefixes=CHECK,CHECK-NOF16
; RUN: %if ptxas %{ llc < %s | %ptxas-verify %}
; RUN: %if ptxas-11.0 %{ llc < %s -mcpu=sm_80 | %ptxas-verify -arch=sm_80 %}
; RUN: %if ptxas-11.0 %{ llc < %s -mcpu=sm_80 --nvptx-no-f16-math | %ptxas-verify -arch=sm_80 %}

target triple = "nvptx64-nvidia-cuda"

; Checks that llvm intrinsics for math functions are correctly lowered to PTX.

declare float @llvm.ceil.f32(float) #0
declare double @llvm.ceil.f64(double) #0
declare float @llvm.floor.f32(float) #0
declare double @llvm.floor.f64(double) #0
declare float @llvm.round.f32(float) #0
declare double @llvm.round.f64(double) #0
declare float @llvm.nearbyint.f32(float) #0
declare double @llvm.nearbyint.f64(double) #0
declare float @llvm.rint.f32(float) #0
declare double @llvm.rint.f64(double) #0
declare float @llvm.trunc.f32(float) #0
declare double @llvm.trunc.f64(double) #0
declare float @llvm.fabs.f32(float) #0
declare double @llvm.fabs.f64(double) #0
declare half @llvm.minnum.f16(half, half) #0
declare float @llvm.minnum.f32(float, float) #0
declare double @llvm.minnum.f64(double, double) #0
declare <2 x half> @llvm.minnum.v2f16(<2 x half>, <2 x half>) #0
declare half @llvm.maxnum.f16(half, half) #0
declare float @llvm.maxnum.f32(float, float) #0
declare double @llvm.maxnum.f64(double, double) #0
declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>) #0
declare float @llvm.fma.f32(float, float, float) #0
declare double @llvm.fma.f64(double, double, double) #0

; ---- ceil ----

; CHECK-LABEL: ceil_float
define float @ceil_float(float %a) {
  ; CHECK: cvt.rpi.f32.f32
  %b = call float @llvm.ceil.f32(float %a)
  ret float %b
}

; CHECK-LABEL: ceil_float_ftz
define float @ceil_float_ftz(float %a) #1 {
  ; CHECK: cvt.rpi.ftz.f32.f32
  %b = call float @llvm.ceil.f32(float %a)
  ret float %b
}

; CHECK-LABEL: ceil_double
define double @ceil_double(double %a) {
  ; CHECK: cvt.rpi.f64.f64
  %b = call double @llvm.ceil.f64(double %a)
  ret double %b
}

; ---- floor ----

; CHECK-LABEL: floor_float
define float @floor_float(float %a) {
  ; CHECK: cvt.rmi.f32.f32
  %b = call float @llvm.floor.f32(float %a)
  ret float %b
}

; CHECK-LABEL: floor_float_ftz
define float @floor_float_ftz(float %a) #1 {
  ; CHECK: cvt.rmi.ftz.f32.f32
  %b = call float @llvm.floor.f32(float %a)
  ret float %b
}

; CHECK-LABEL: floor_double
define double @floor_double(double %a) {
  ; CHECK: cvt.rmi.f64.f64
  %b = call double @llvm.floor.f64(double %a)
  ret double %b
}

; ---- round ----

; CHECK-LABEL: round_float
define float @round_float(float %a) {
; check the use of sign mask and 0.5 to implement round
; CHECK: and.b32 [[R1:%r[0-9]+]], {{.*}}, -2147483648;
; CHECK: or.b32 {{.*}}, [[R1]], 1056964608;
  %b = call float @llvm.round.f32(float %a)
  ret float %b
}

; CHECK-LABEL: round_float_ftz
define float @round_float_ftz(float %a) #1 {
; check the use of sign mask and 0.5 to implement round
; CHECK: and.b32 [[R1:%r[0-9]+]], {{.*}}, -2147483648;
; CHECK: or.b32 {{.*}}, [[R1]], 1056964608;
  %b = call float @llvm.round.f32(float %a)
  ret float %b
}

; CHECK-LABEL: round_double
define double @round_double(double %a) {
; check the use of 0.5 to implement round
; CHECK: setp.lt.f64 {{.*}}, [[R:%fd[0-9]+]], 0d3FE0000000000000;
; CHECK: add.rn.f64 {{.*}}, [[R]], 0d3FE0000000000000;
  %b = call double @llvm.round.f64(double %a)
  ret double %b
}

; ---- nearbyint ----

; CHECK-LABEL: nearbyint_float
define float @nearbyint_float(float %a) {
  ; CHECK: cvt.rni.f32.f32
  %b = call float @llvm.nearbyint.f32(float %a)
  ret float %b
}

; CHECK-LABEL: nearbyint_float_ftz
define float @nearbyint_float_ftz(float %a) #1 {
  ; CHECK: cvt.rni.ftz.f32.f32
  %b = call float @llvm.nearbyint.f32(float %a)
  ret float %b
}

; CHECK-LABEL: nearbyint_double
define double @nearbyint_double(double %a) {
  ; CHECK: cvt.rni.f64.f64
  %b = call double @llvm.nearbyint.f64(double %a)
  ret double %b
}

; ---- rint ----

; CHECK-LABEL: rint_float
define float @rint_float(float %a) {
  ; CHECK: cvt.rni.f32.f32
  %b = call float @llvm.rint.f32(float %a)
  ret float %b
}

; CHECK-LABEL: rint_float_ftz
define float @rint_float_ftz(float %a) #1 {
  ; CHECK: cvt.rni.ftz.f32.f32
  %b = call float @llvm.rint.f32(float %a)
  ret float %b
}

; CHECK-LABEL: rint_double
define double @rint_double(double %a) {
  ; CHECK: cvt.rni.f64.f64
  %b = call double @llvm.rint.f64(double %a)
  ret double %b
}

; ---- trunc ----

; CHECK-LABEL: trunc_float
define float @trunc_float(float %a) {
  ; CHECK: cvt.rzi.f32.f32
  %b = call float @llvm.trunc.f32(float %a)
  ret float %b
}

; CHECK-LABEL: trunc_float_ftz
define float @trunc_float_ftz(float %a) #1 {
  ; CHECK: cvt.rzi.ftz.f32.f32
  %b = call float @llvm.trunc.f32(float %a)
  ret float %b
}

; CHECK-LABEL: trunc_double
define double @trunc_double(double %a) {
  ; CHECK: cvt.rzi.f64.f64
  %b = call double @llvm.trunc.f64(double %a)
  ret double %b
}

; ---- abs ----

; CHECK-LABEL: abs_float
define float @abs_float(float %a) {
  ; CHECK: abs.f32
  %b = call float @llvm.fabs.f32(float %a)
  ret float %b
}

; CHECK-LABEL: abs_float_ftz
define float @abs_float_ftz(float %a) #1 {
  ; CHECK: abs.ftz.f32
  %b = call float @llvm.fabs.f32(float %a)
  ret float %b
}

; CHECK-LABEL: abs_double
define double @abs_double(double %a) {
  ; CHECK: abs.f64
  %b = call double @llvm.fabs.f64(double %a)
  ret double %b
}

; ---- min ----

; CHECK-LABEL: min_half
define half @min_half(half %a, half %b) {
  ; CHECK-NOF16: min.f32
  ; CHECK-F16: min.f16
  %x = call half @llvm.minnum.f16(half %a, half %b)
  ret half %x
}

; CHECK-LABEL: min_float
define float @min_float(float %a, float %b) {
  ; CHECK: min.f32
  %x = call float @llvm.minnum.f32(float %a, float %b)
  ret float %x
}

; CHECK-LABEL: min_imm1
define float @min_imm1(float %a) {
  ; CHECK: min.f32
  %x = call float @llvm.minnum.f32(float %a, float 0.0)
  ret float %x
}

; CHECK-LABEL: min_imm2
define float @min_imm2(float %a) {
  ; CHECK: min.f32
  %x = call float @llvm.minnum.f32(float 0.0, float %a)
  ret float %x
}

; CHECK-LABEL: min_float_ftz
define float @min_float_ftz(float %a, float %b) #1 {
  ; CHECK: min.ftz.f32
  %x = call float @llvm.minnum.f32(float %a, float %b)
  ret float %x
}

; CHECK-LABEL: min_double
define double @min_double(double %a, double %b) {
  ; CHECK: min.f64
  %x = call double @llvm.minnum.f64(double %a, double %b)
  ret double %x
}

; CHECK-LABEL: min_v2half
define <2 x half> @min_v2half(<2 x half> %a, <2 x half> %b) {
  ; CHECK-NOF16: min.f32
  ; CHECK-NOF16: min.f32
  ; CHECK-F16: min.f16x2
  %x = call <2 x half> @llvm.minnum.v2f16(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %x
}

; ---- max ----

; CHECK-LABEL: max_half
define half @max_half(half %a, half %b) {
  ; CHECK-NOF16: max.f32
  ; CHECK-F16: max.f16
  %x = call half @llvm.maxnum.f16(half %a, half %b)
  ret half %x
}

; CHECK-LABEL: max_imm1
define float @max_imm1(float %a) {
  ; CHECK: max.f32
  %x = call float @llvm.maxnum.f32(float %a, float 0.0)
  ret float %x
}

; CHECK-LABEL: max_imm2
define float @max_imm2(float %a) {
  ; CHECK: max.f32
  %x = call float @llvm.maxnum.f32(float 0.0, float %a)
  ret float %x
}

; CHECK-LABEL: max_float
define float @max_float(float %a, float %b) {
  ; CHECK: max.f32
  %x = call float @llvm.maxnum.f32(float %a, float %b)
  ret float %x
}

; CHECK-LABEL: max_float_ftz
define float @max_float_ftz(float %a, float %b) #1 {
  ; CHECK: max.ftz.f32
  %x = call float @llvm.maxnum.f32(float %a, float %b)
  ret float %x
}

; CHECK-LABEL: max_double
define double @max_double(double %a, double %b) {
  ; CHECK: max.f64
  %x = call double @llvm.maxnum.f64(double %a, double %b)
  ret double %x
}

; CHECK-LABEL: max_v2half
define <2 x half> @max_v2half(<2 x half> %a, <2 x half> %b) {
  ; CHECK-NOF16: max.f32
  ; CHECK-NOF16: max.f32
  ; CHECK-F16: max.f16x2
  %x = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %a, <2 x half> %b)
  ret <2 x half> %x
}

; ---- fma ----

; CHECK-LABEL: @fma_float
define float @fma_float(float %a, float %b, float %c) {
  ; CHECK: fma.rn.f32
  %x = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %x
}

; CHECK-LABEL: @fma_float_ftz
define float @fma_float_ftz(float %a, float %b, float %c) #1 {
  ; CHECK: fma.rn.ftz.f32
  %x = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %x
}

; CHECK-LABEL: @fma_double
define double @fma_double(double %a, double %b, double %c) {
  ; CHECK: fma.rn.f64
  %x = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %x
}

attributes #0 = { nounwind readnone }
attributes #1 = { "denormal-fp-math-f32" = "preserve-sign" }
