// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -convert-gpu-to-spirv -verify-diagnostics %s -o - | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @basic_module_structure
    // CHECK-SAME: {{%.*}}: f32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>}
    // CHECK-SAME: {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @basic_module_structure(%arg0 : f32, %arg1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: spirv.Return
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32, #spirv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@basic_module_structure
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32, %1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>)
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-LABEL: spirv.func @basic_module_structure_preset_ABI
    // CHECK-SAME: {{%[a-zA-Z0-9_]*}}: f32
    // CHECK-SAME: spirv.interface_var_abi = #spirv.interface_var_abi<(1, 2), StorageBuffer>
    // CHECK-SAME: !spirv.ptr<!spirv.struct<(!spirv.array<12 x f32, stride=4> [0])>, StorageBuffer>
    // CHECK-SAME: spirv.interface_var_abi = #spirv.interface_var_abi<(3, 0)>
    // CHECK-SAME: spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>
    gpu.func @basic_module_structure_preset_ABI(
      %arg0 : f32
        {spirv.interface_var_abi = #spirv.interface_var_abi<(1, 2), StorageBuffer>},
      %arg1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>
        {spirv.interface_var_abi = #spirv.interface_var_abi<(3, 0)>}) kernel
      attributes
        {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: spirv.Return
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-remark @below {{match failure: missing 'spirv.entry_point_abi' attribute}}
    gpu.func @missing_entry_point_abi(%arg0 : f32, %arg1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>) kernel {
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32, #spirv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@missing_entry_point_abi
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32, %1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>)
    return
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-remark @below {{match failure: missing 'spirv.interface_var_abi' attribute at argument 1}}
    gpu.func @missing_entry_point_abi(
      %arg0 : f32
        {spirv.interface_var_abi = #spirv.interface_var_abi<(1, 2), StorageBuffer>},
      %arg1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>) kernel
    attributes
      {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-remark @below {{match failure: missing 'spirv.interface_var_abi' attribute at argument 0}}
    gpu.func @missing_entry_point_abi(
      %arg0 : f32,
      %arg1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>
        {spirv.interface_var_abi = #spirv.interface_var_abi<(3, 0)>}) kernel
    attributes
      {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      gpu.return
    }
  }
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    // expected-error @below {{failed to legalize operation 'gpu.func'}}
    // expected-error @below {{SPIR-V lowering of private attributions is not supported}}
    gpu.func @private_attribution_unsupported(
      %arg0: memref<256xf32, #spirv.storage_class<StorageBuffer>>)
      workgroup(%wg: memref<256xf32, #spirv.storage_class<Workgroup>>)
      private(%priv: memref<4xf32, #spirv.storage_class<Function>>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>} {
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK-LABEL: spirv.func @barrier
    gpu.func @barrier(%arg0 : f32, %arg1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [32, 4, 1]>} {
      // CHECK: spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
      gpu.barrier
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<12xf32, #spirv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@barrier
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32, %1 : memref<12xf32, #spirv.storage_class<StorageBuffer>>)
    return
  }
}

// -----

// Test gpu.func with a single workgroup attribution.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-DAG:   spirv.GlobalVariable @__workgroup_mem__kernel_wg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32>)>, Workgroup>
    // CHECK-LABEL: spirv.func @kernel_wg
    // CHECK-SAME:  {{%.*}}: f32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0), StorageBuffer>}
    // CHECK-NOT:   Workgroup
    // CHECK-SAME:  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>
    // CHECK:       spirv.mlir.addressof @__workgroup_mem__kernel_wg_0
    // CHECK:       spirv.Return
    gpu.func @kernel_wg(%arg0 : f32)
      workgroup(%wg : memref<256xf32, #spirv.storage_class<Workgroup>>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>} {
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (f32)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@kernel_wg
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : f32)
    return
  }
}

// -----

// Test gpu.func with multiple workgroup attributions.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-DAG:   spirv.GlobalVariable @__workgroup_mem__kernel_multi_wg_0 : !spirv.ptr<!spirv.struct<(!spirv.array<128 x f32>)>, Workgroup>
    // CHECK-DAG:   spirv.GlobalVariable @__workgroup_mem__kernel_multi_wg_1 : !spirv.ptr<!spirv.struct<(!spirv.array<64 x i32>)>, Workgroup>
    // CHECK-LABEL: spirv.func @kernel_multi_wg
    // CHECK-SAME:  {{%.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME:  spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [128, 1, 1]>
    // CHECK:       spirv.mlir.addressof @__workgroup_mem__kernel_multi_wg_0
    // CHECK:       spirv.mlir.addressof @__workgroup_mem__kernel_multi_wg_1
    // CHECK:       spirv.Return
    gpu.func @kernel_multi_wg(
      %arg0: memref<256xf32, #spirv.storage_class<StorageBuffer>>)
      workgroup(
        %wg0: memref<128xf32, #spirv.storage_class<Workgroup>>,
        %wg1: memref<64xi32, #spirv.storage_class<Workgroup>>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [128, 1, 1]>} {
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (memref<256xf32, #spirv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@kernel_multi_wg
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : memref<256xf32, #spirv.storage_class<StorageBuffer>>)
    return
  }
}

// -----

// Test gpu.func with workgroup attribution and barrier.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK:       spirv.module @{{.*}} Logical GLSL450 {
    // CHECK-DAG:   spirv.GlobalVariable @__workgroup_mem__kernel_wg_barrier_0 : !spirv.ptr<!spirv.struct<(!spirv.array<256 x f32>)>, Workgroup>
    // CHECK-LABEL: spirv.func @kernel_wg_barrier
    // CHECK:       spirv.mlir.addressof @__workgroup_mem__kernel_wg_barrier_0
    // CHECK:       spirv.ControlBarrier <Workgroup>, <Workgroup>, <AcquireRelease|WorkgroupMemory>
    // CHECK:       spirv.Return
    gpu.func @kernel_wg_barrier(
      %arg0: memref<256xf32, #spirv.storage_class<StorageBuffer>>)
      workgroup(%wg: memref<256xf32, #spirv.storage_class<Workgroup>>)
      kernel attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [256, 1, 1]>} {
      gpu.barrier
      gpu.return
    }
  }

  func.func @main() {
    %0 = "op"() : () -> (memref<256xf32, #spirv.storage_class<StorageBuffer>>)
    %cst = arith.constant 1 : index
    gpu.launch_func @kernels::@kernel_wg_barrier
        blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)
        args(%0 : memref<256xf32, #spirv.storage_class<StorageBuffer>>)
    return
  }
}
