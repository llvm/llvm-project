// RUN: mlir-opt %s --loop-invariant-code-motion | FileCheck %s

func.func private @side_effect(%arg0 : f32, %arg1 : f32)

// CHECK-LABEL: func @assume_subgroup_uniform_hoisting
//  CHECK-SAME: (%[[ARG:.*]]: f32)
func.func @assume_subgroup_uniform_hoisting(%arg0 : f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
// CHECK: %[[V1:.*]] = amdgpu.assume_subgroup_uniform all_lanes %[[ARG]] : f32
// CHECK: scf.for
// CHECK: %[[V0:.*]] = amdgpu.assume_subgroup_uniform %[[ARG]] : f32
// CHECK: func.call @side_effect(%[[V0]], %[[V1]])
  scf.for %i = %c0 to %c10 step %c1 {
    %0 = amdgpu.assume_subgroup_uniform %arg0 : f32
    %1 = amdgpu.assume_subgroup_uniform all_lanes %arg0 : f32
    func.call @side_effect(%0, %1) : (f32, f32) -> ()
  }
  func.return
}
