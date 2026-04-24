// RUN: mlir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @nvvm_cos_f32
func.func @nvvm_cos_f32(%arg0: f32) -> f32 {
  // CHECK: nvvm.cos {{.*}} : f32
  %0 = nvvm.cos %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @nvvm_cos_ftz_f32
func.func @nvvm_cos_ftz_f32(%arg0: f32) -> f32 {
  // CHECK: nvvm.cos {{.*}} {ftz = true} : f32
  %0 = nvvm.cos %arg0 {ftz = true} : f32
  return %0 : f32
}
// RUN: mlir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @nvvm_sin_f32
func.func @nvvm_sin_f32(%arg0: f32) -> f32 {
  // CHECK: nvvm.sin {{.*}} : f32
  %0 = nvvm.sin %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @nvvm_sin_ftz_f32
func.func @nvvm_sin_ftz_f32(%arg0: f32) -> f32 {
  // CHECK: nvvm.sin {{.*}} {ftz = true} : f32
  %0 = nvvm.sin %arg0 {ftz = true} : f32
  return %0 : f32
}

// CHECK-LABEL: @nvvm_ex2_f32
func.func @nvvm_ex2_f32(%arg0: f32) -> f32 {
  // CHECK: nvvm.ex2 {{.*}} : f32
  %0 = nvvm.ex2 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: @nvvm_ex2_ftz_f32
func.func @nvvm_ex2_ftz_f32(%arg0: f32) -> f32 {
  // CHECK: nvvm.ex2 {{.*}} {ftz = true} : f32
  %0 = nvvm.ex2 %arg0 {ftz = true} : f32
  return %0 : f32
}
