; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define float @f32(float %a, i32 %b) {
  ; CHECK: %call = call float @llvm.ldexp.f32.i32(float %a, i32 %b)
  ; CHECK-NOT: amdgcn.ldexp
  %call = call float @llvm.amdgcn.ldexp.f32(float %a, i32 %b)
  ret float %call
}

define double @f64(double %a, i32 %b) {
  ; CHECK: %call = call double @llvm.ldexp.f64.i32(double %a, i32 %b)
  ; CHECK-NOT: amdgcn.ldexp
  %call = call double @llvm.amdgcn.ldexp.f64(double %a, i32 %b)
  ret double %call
}

define half @f16(half %a, i32 %b) {
  ; CHECK: %call = call half @llvm.ldexp.f16.i32(half %a, i32 %b)
  ; CHECK-NOT: amdgcn.ldexp
  %call = call half @llvm.amdgcn.ldexp.f16(half %a, i32 %b)
  ret half %call
}

declare half @llvm.amdgcn.ldexp.f16(half, i32)
declare float @llvm.amdgcn.ldexp.f32(float, i32)
declare double @llvm.amdgcn.ldexp.f64(double, i32)
; CHECK: declare half @llvm.ldexp.f16.i32(half, i32)
; CHECK: declare float @llvm.ldexp.f32.i32(float, i32)
; CHECK: declare double @llvm.ldexp.f64.i32(double, i32)
; CHECK-NOT: amdgcn.ldexp
