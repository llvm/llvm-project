// RUN: mlir-opt -convert-to-spirv="convert-gpu-modules=true run-signature-conversion=false run-vector-unrolling=false" -split-input-file %s | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], []>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: func.func @main
  // CHECK:       %[[C1:.*]] = arith.constant 1 : index
  // CHECK:       gpu.launch_func  @[[$KERNELS_1:.*]]::@[[$BUILTIN_WG_ID_X:.*]] blocks in (%[[C1]], %[[C1]], %[[C1]]) threads in (%[[C1]], %[[C1]], %[[C1]])
  // CHECK:       gpu.launch_func  @[[$KERNELS_2:.*]]::@[[$BUILTIN_WG_ID_Y:.*]] blocks in (%[[C1]], %[[C1]], %[[C1]]) threads in (%[[C1]], %[[C1]], %[[C1]])
  func.func @main() {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels_1::@builtin_workgroup_id_x
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
    gpu.launch_func @KERNELS_2::@builtin_workgroup_id_y
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
    return
  }

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK:        spirv.func @[[$BUILTIN_WG_ID_X]]
  // CHECK:        spirv.mlir.addressof
  // CHECK:        spirv.Load "Input"
  // CHECK:        spirv.CompositeExtract
  gpu.module @kernels_1 {
    gpu.func @builtin_workgroup_id_x() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = gpu.block_id x
      gpu.return
    }
  }
  // CHECK:  gpu.module @[[$KERNELS_1]]
  // CHECK:  gpu.func @[[$BUILTIN_WG_ID_X]]
  // CHECK   gpu.block_id x
  // CHECK:  gpu.return

  // CHECK-LABEL:  spirv.module @{{.*}} Logical GLSL450
  // CHECK:        spirv.func @[[$BUILTIN_WG_ID_Y]]
  // CHECK:        spirv.mlir.addressof
  // CHECK:        spirv.Load "Input"
  // CHECK:        spirv.CompositeExtract
  gpu.module @KERNELS_2 {
    gpu.func @builtin_workgroup_id_y() kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [16, 1, 1]>} {
      %0 = gpu.block_id y
      gpu.return
    }
  }
  // CHECK:  gpu.module @[[$KERNELS_2]]
  // CHECK:  gpu.func @[[$BUILTIN_WG_ID_Y]]
  // CHECK   gpu.block_id y
  // CHECK:  gpu.return
}

// -----

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  // CHECK-LABEL: func.func @main
  // CHECK-SAME:  %[[ARG0:.*]]: memref<2xi32>, %[[ARG1:.*]]: memref<4xi32>
  // CHECK:       %[[C1:.*]] = arith.constant 1 : index
  // CHECK:       gpu.launch_func  @[[$KERNEL_MODULE:.*]]::@[[$KERNEL_FUNC:.*]] blocks in (%[[C1]], %[[C1]], %[[C1]]) threads in (%[[C1]], %[[C1]], %[[C1]]) args(%[[ARG0]] : memref<2xi32>, %[[ARG1]] : memref<4xi32>)
  func.func @main(%arg0 : memref<2xi32>, %arg2 : memref<4xi32>) {
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@kernel_foo
        blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
        args(%arg0 : memref<2xi32>, %arg2 : memref<4xi32>)
    return
  }

  // CHECK-LABEL: spirv.module @{{.*}} Logical GLSL450
  // CHECK:       spirv.func @[[$KERNEL_FUNC]]
  // CHECK-SAME:  %{{.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<2 x i32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
  // CHECK-SAME:  %{{.*}}: !spirv.ptr<!spirv.struct<(!spirv.array<4 x i32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
  gpu.module @kernels {
    gpu.func @kernel_foo(%arg0 : memref<2xi32>, %arg1 : memref<4xi32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      // CHECK: spirv.Constant
      // CHECK: spirv.Constant dense<0>
      %idx0 = arith.constant 0 : index
      %vec0 = arith.constant dense<[0, 0]> : vector<2xi32>
      // CHECK: spirv.AccessChain
      // CHECK: spirv.Load "StorageBuffer"
      %val = memref.load %arg0[%idx0] : memref<2xi32>
      // CHECK: spirv.CompositeInsert
      %vec = vector.insertelement %val, %vec0[%idx0 : index] : vector<2xi32>
      // CHECK: spirv.VectorShuffle
      %shuffle = vector.shuffle %vec, %vec[3, 2, 1, 0] : vector<2xi32>, vector<2xi32>
      // CHECK: spirv.CompositeExtract
      %res = vector.extractelement %shuffle[%idx0 : index] : vector<4xi32>
      // CHECK: spirv.AccessChain
      // CHECK: spirv.Store "StorageBuffer"
      memref.store %res, %arg1[%idx0]: memref<4xi32>
      // CHECK: spirv.Return
      gpu.return
    }
  }
  // CHECK:      gpu.module @[[$KERNEL_MODULE]]
  // CHECK:      gpu.func @[[$KERNEL_FUNC]]
  // CHECK-SAME: %{{.*}}: memref<2xi32>, %{{.*}}: memref<4xi32>
  // CHECK:      arith.constant
  // CHECK:      memref.load
  // CHECK:      vector.insertelement
  // CHECK:      vector.shuffle
  // CHECK:      vector.extractelement
  // CHECK:      memref.store
  // CHECK:      gpu.return
}
