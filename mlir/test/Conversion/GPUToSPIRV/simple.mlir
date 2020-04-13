// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spv.module Logical GLSL450 {
    // CHECK-LABEL: spv.func @basic_module_structure
    // CHECK-SAME: {{%.*}}: f32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 0), StorageBuffer>}
    // CHECK-SAME: {{%.*}}: !spv.ptr<!spv.struct<!spv.array<12 x f32, stride=4> [0]>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}
    gpu.func @basic_module_structure(%arg0 : f32, %arg1 : memref<12xf32>)
      attributes {gpu.kernel, spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // CHECK: spv.Return
      gpu.return
    }
  }

  func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32>)
    %cst = constant 1 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = "basic_module_structure", kernel_module = @kernels }
        : (index, index, index, index, index, index, f32, memref<12xf32>) -> ()
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-remark @below {{match failure: missing 'spv.entry_point_abi' attribute}}
    gpu.func @missing_entry_point_abi(%arg0 : f32, %arg1 : memref<12xf32>) attributes {gpu.kernel} {
      gpu.return
    }
  }

  func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32>)
    %cst = constant 1 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %0, %1) { kernel = "missing_entry_point_abi", kernel_module = @kernels }
        : (index, index, index, index, index, index, f32, memref<12xf32>) -> ()
    return
  }
}
