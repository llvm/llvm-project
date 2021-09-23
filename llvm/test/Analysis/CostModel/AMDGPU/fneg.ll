; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s
; RUN: opt -cost-model -cost-kind=code-size -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s
; END.

; CHECK-LABEL: 'fneg_f32'
; CHECK: estimated cost of 0 for {{.*}} fneg float
; CHECK: estimated cost of 0 for {{.*}} fneg <2 x float>
; CHECK: estimated cost of 0 for {{.*}} fneg <3 x float>
; CHECK: estimated cost of 0 for {{.*}} fneg <5 x float>
define amdgpu_kernel void @fneg_f32() {
  %f32 = fneg float undef
  %v2f32 = fneg <2 x float> undef
  %v3f32 = fneg <3 x float> undef
  %v5f32 = fneg <5 x float> undef
  ret void
}

; CHECK-LABEL: 'fneg_f64'
; CHECK: estimated cost of 0 for {{.*}} fneg double
; CHECK: estimated cost of 0 for {{.*}} fneg <2 x double>
; CHECK: estimated cost of 0 for {{.*}} fneg <3 x double>
define amdgpu_kernel void @fneg_f64() {
  %f64 = fneg double undef
  %v2f64 = fneg <2 x double> undef
  %v3f64 = fneg <3 x double> undef
  ret void
}

; CHECK-LABEL: 'fneg_f16'
; CHECK: estimated cost of 0 for {{.*}} fneg half
; CHECK: estimated cost of 0 for {{.*}} fneg <2 x half>
; CHECK: estimated cost of 0 for {{.*}} fneg <3 x half>
define amdgpu_kernel void @fneg_f16() {
  %f16 = fneg half undef
  %v2f16 = fneg <2 x half> undef
  %v3f16 = fneg <3 x half> undef
  ret void
}
