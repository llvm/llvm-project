// RUN: mlir-opt %s --inline | FileCheck %s

func.func @calee(%arg0 : f32) -> f32 {
  %0 = amdgpu.permlane_swap %arg0 32 : f32
  func.return %0 : f32
}

// CHECK-LABEL: func @caller
func.func @caller(%arg0 : f32) -> f32 {
  // CHECK-NOT: call
  //     CHECK: amdgpu.permlane_swap
  %0 = call @calee(%arg0) : (f32) -> f32
  func.return %0 : f32
}
