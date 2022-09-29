// RUN: mlir-opt -allow-unregistered-dialect -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel, Addresses], []>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    // CHECK-LABEL: spirv.module @{{.*}} Physical64 OpenCL
    //       CHECK:   spirv.func
    //  CHECK-SAME:     {{%.*}}: f32
    //   CHECK-NOT:     spirv.interface_var_abi
    //  CHECK-SAME:     {{%.*}}: !spirv.ptr<!spirv.array<12 x f32>, CrossWorkgroup>
    //   CHECK-NOT:     spirv.interface_var_abi
    //  CHECK-SAME:     spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 4, 1]> : vector<3xi32>>
    gpu.func @basic_module_structure(%arg0 : f32, %arg1 : memref<12xf32, #spirv.storage_class<CrossWorkgroup>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[32, 4, 1]>: vector<3xi32>>} {
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32, #spirv.storage_class<CrossWorkgroup>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@basic_module_structure
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32, %1 : memref<12xf32, #spirv.storage_class<CrossWorkgroup>>)
    return
  }
}
