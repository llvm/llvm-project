// RUN: mlir-opt -allow-unregistered-dialect -convert-gpu-to-spirv="use-64bit-index=true" -verify-diagnostics -split-input-file %s -o - | FileCheck %s

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
    //  CHECK-SAME:     spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @basic_module_structure(%arg0 : f32, %arg1 : memref<12xf32, #spirv.storage_class<CrossWorkgroup>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
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

// -----

module attributes {
  gpu.container_module
} {
  gpu.module @kernels attributes {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel, Addresses], []>, #spirv.resource_limits<>>
  } {
    // CHECK-LABEL: spirv.module @{{.*}} Physical64 OpenCL
    //  CHECK-SAME: spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Kernel, Addresses], []>, #spirv.resource_limits<>>
    //       CHECK:   spirv.func
    //  CHECK-SAME:     {{%.*}}: f32
    //   CHECK-NOT:     spirv.interface_var_abi
    //  CHECK-SAME:     {{%.*}}: !spirv.ptr<!spirv.array<12 x f32>, CrossWorkgroup>
    //   CHECK-NOT:     spirv.interface_var_abi
    //  CHECK-SAME:     spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @basic_module_structure(%arg0 : f32, %arg1 : memref<12xf32, #spirv.storage_class<CrossWorkgroup>>) kernel
        attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
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
