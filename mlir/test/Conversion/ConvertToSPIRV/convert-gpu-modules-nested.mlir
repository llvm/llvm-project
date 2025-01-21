// RUN: mlir-opt -convert-to-spirv="convert-gpu-modules=true nest-in-gpu-module=true run-signature-conversion=false run-vector-unrolling=false" %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: func.func @main
  // CHECK:       %[[C1:.*]] = arith.constant 1 : index
  // CHECK:       gpu.launch_func  @[[$KERNELS_1:.*]]::@[[$BUILTIN_WG_ID_X:.*]] blocks in (%[[C1]], %[[C1]], %[[C1]]) threads in (%[[C1]], %[[C1]], %[[C1]])
  func.func @main() {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels_1::@builtin_workgroup_id_x
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
    return
  }

  // CHECK: gpu.module @[[$KERNELS_1]]
  // CHECK:   spirv.module @{{.*}} Logical GLSL450
  // CHECK:   spirv.func @[[$BUILTIN_WG_ID_X]]
  // CHECK:   spirv.mlir.addressof
  // CHECK:   spirv.Load "Input"
  // CHECK:   spirv.CompositeExtract
  gpu.module @kernels_1 {
    gpu.func @builtin_workgroup_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = gpu.block_id x
      gpu.return
    }
  }
}
