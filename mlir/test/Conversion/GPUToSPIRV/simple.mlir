// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spv.func @basic_module_structure
    // CHECK-SAME: {{%.*}}: f32 {spv.interface_var_abi = #spv.interface_var_abi<(0, 0), StorageBuffer>}
    // CHECK-SAME: {{%.*}}: !spv.ptr<!spv.struct<(!spv.array<12 x f32, stride=4> [0])>, StorageBuffer> {spv.interface_var_abi = #spv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}
    gpu.func @basic_module_structure(%arg0 : f32, %arg1 : memref<12xf32>) kernel
      attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // CHECK: spv.Return
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@basic_module_structure
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32, %1 : memref<12xf32>)
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spv.func @basic_module_structure_preset_ABI
    // CHECK-SAME: {{%[a-zA-Z0-9_]*}}: f32
    // CHECK-SAME: spv.interface_var_abi = #spv.interface_var_abi<(1, 2), StorageBuffer>
    // CHECK-SAME: !spv.ptr<!spv.struct<(!spv.array<12 x f32, stride=4> [0])>, StorageBuffer>
    // CHECK-SAME: spv.interface_var_abi = #spv.interface_var_abi<(3, 0)>
    // CHECK-SAME: spv.entry_point_abi = {local_size = dense<[32, 4, 1]> : vector<3xi32>}
    gpu.func @basic_module_structure_preset_ABI(
      %arg0 : f32
        {spv.interface_var_abi = #spv.interface_var_abi<(1, 2), StorageBuffer>},
      %arg1 : memref<12xf32>
        {spv.interface_var_abi = #spv.interface_var_abi<(3, 0)>}) kernel
      attributes
        {spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // CHECK: spv.Return
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-remark @below {{match failure: missing 'spv.entry_point_abi' attribute}}
    gpu.func @missing_entry_point_abi(%arg0 : f32, %arg1 : memref<12xf32>) kernel {
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@missing_entry_point_abi
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32, %1 : memref<12xf32>)
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-remark @below {{match failure: missing 'spv.interface_var_abi' attribute at argument 1}}
    gpu.func @missing_entry_point_abi(
      %arg0 : f32
        {spv.interface_var_abi = #spv.interface_var_abi<(1, 2), StorageBuffer>},
      %arg1 : memref<12xf32>) kernel
    attributes
      {spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-remark @below {{match failure: missing 'spv.interface_var_abi' attribute at argument 0}}
    gpu.func @missing_entry_point_abi(
      %arg0 : f32,
      %arg1 : memref<12xf32>
        {spv.interface_var_abi = #spv.interface_var_abi<(3, 0)>}) kernel
    attributes
      {spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK-LABEL: spv.func @barrier
    gpu.func @barrier(%arg0 : f32, %arg1 : memref<12xf32>) kernel
      attributes {spv.entry_point_abi = {local_size = dense<[32, 4, 1]>: vector<3xi32>}} {
      // CHECK: spv.ControlBarrier Workgroup, Workgroup, "AcquireRelease|WorkgroupMemory"
      gpu.barrier
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@barrier
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32, %1 : memref<12xf32>)
    return
  }
}
