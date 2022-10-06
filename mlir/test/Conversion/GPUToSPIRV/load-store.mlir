// RUN: mlir-opt -convert-gpu-to-spirv %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  func.func @load_store(%arg0: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg2: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>) {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %0 = arith.subi %c12, %c0 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %1 = arith.subi %c4, %c0_0 : index
    %c1_1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    gpu.launch_func @kernels::@load_store_kernel
        blocks in (%0, %c1_2, %c1_2) threads in (%1, %c1_2, %c1_2)
        args(%arg0 : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg1 : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg2 : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>,
             %c0 : index, %c0_0 : index, %c1 : index, %c1_1 : index)
    return
  }

  // CHECK-LABEL: spirv.module @{{.*}} Logical GLSL450
  gpu.module @kernels {
    // CHECK-DAG: spirv.GlobalVariable @[[NUMWORKGROUPSVAR:.*]] built_in("NumWorkgroups") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG: spirv.GlobalVariable @[[$LOCALINVOCATIONIDVAR:.*]] built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-DAG: spirv.GlobalVariable @[[$WORKGROUPIDVAR:.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
    // CHECK-LABEL:    spirv.func @load_store_kernel
    // CHECK-SAME: %[[ARG0:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 0)>}
    // CHECK-SAME: %[[ARG1:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 1)>}
    // CHECK-SAME: %[[ARG2:.*]]: !spirv.ptr<!spirv.struct<(!spirv.array<48 x f32, stride=4> [0])>, StorageBuffer> {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 2)>}
    // CHECK-SAME: %[[ARG3:.*]]: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 3), StorageBuffer>}
    // CHECK-SAME: %[[ARG4:.*]]: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 4), StorageBuffer>}
    // CHECK-SAME: %[[ARG5:.*]]: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 5), StorageBuffer>}
    // CHECK-SAME: %[[ARG6:.*]]: i32 {spirv.interface_var_abi = #spirv.interface_var_abi<(0, 6), StorageBuffer>}
    gpu.func @load_store_kernel(%arg0: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg1: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg2: memref<12x4xf32, #spirv.storage_class<StorageBuffer>>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) kernel
      attributes {spirv.entry_point_abi = #spirv.entry_point_abi<local_size = dense<[16, 1, 1]>: vector<3xi32>>} {
      // CHECK: %[[ADDRESSWORKGROUPID:.*]] = spirv.mlir.addressof @[[$WORKGROUPIDVAR]]
      // CHECK: %[[WORKGROUPID:.*]] = spirv.Load "Input" %[[ADDRESSWORKGROUPID]]
      // CHECK: %[[WORKGROUPIDX:.*]] = spirv.CompositeExtract %[[WORKGROUPID]]{{\[}}0 : i32{{\]}}
      // CHECK: %[[ADDRESSLOCALINVOCATIONID:.*]] = spirv.mlir.addressof @[[$LOCALINVOCATIONIDVAR]]
      // CHECK: %[[LOCALINVOCATIONID:.*]] = spirv.Load "Input" %[[ADDRESSLOCALINVOCATIONID]]
      // CHECK: %[[LOCALINVOCATIONIDX:.*]] = spirv.CompositeExtract %[[LOCALINVOCATIONID]]{{\[}}0 : i32{{\]}}
      %0 = gpu.block_id x
      %1 = gpu.block_id y
      %2 = gpu.block_id z
      %3 = gpu.thread_id x
      %4 = gpu.thread_id y
      %5 = gpu.thread_id z
      %6 = gpu.grid_dim x
      %7 = gpu.grid_dim y
      %8 = gpu.grid_dim z
      %9 = gpu.block_dim x
      %10 = gpu.block_dim y
      %11 = gpu.block_dim z
      // CHECK: %[[INDEX1:.*]] = spirv.IAdd %[[ARG3]], %[[WORKGROUPIDX]]
      %12 = arith.addi %arg3, %0 : index
      // CHECK: %[[INDEX2:.*]] = spirv.IAdd %[[ARG4]], %[[LOCALINVOCATIONIDX]]
      %13 = arith.addi %arg4, %3 : index
      // CHECK: %[[ZERO:.*]] = spirv.Constant 0 : i32
      // CHECK: %[[OFFSET1_0:.*]] = spirv.Constant 0 : i32
      // CHECK: %[[STRIDE1_1:.*]] = spirv.Constant 4 : i32
      // CHECK: %[[UPDATE1_1:.*]] = spirv.IMul %[[STRIDE1_1]], %[[INDEX1]] : i32
      // CHECK: %[[OFFSET1_1:.*]] = spirv.IAdd %[[OFFSET1_0]], %[[UPDATE1_1]] : i32
      // CHECK: %[[STRIDE1_2:.*]] = spirv.Constant 1 : i32
      // CHECK: %[[UPDATE1_2:.*]] = spirv.IMul %[[STRIDE1_2]], %[[INDEX2]] : i32
      // CHECK: %[[OFFSET1_2:.*]] = spirv.IAdd %[[OFFSET1_1]], %[[UPDATE1_2]] : i32
      // CHECK: %[[PTR1:.*]] = spirv.AccessChain %[[ARG0]]{{\[}}%[[ZERO]], %[[OFFSET1_2]]{{\]}}
      // CHECK-NEXT: %[[VAL1:.*]] = spirv.Load "StorageBuffer" %[[PTR1]]
      %14 = memref.load %arg0[%12, %13] : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>
      // CHECK: %[[PTR2:.*]] = spirv.AccessChain %[[ARG1]]{{\[}}{{%.*}}, {{%.*}}{{\]}}
      // CHECK-NEXT: %[[VAL2:.*]] = spirv.Load "StorageBuffer" %[[PTR2]]
      %15 = memref.load %arg1[%12, %13] : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>
      // CHECK: %[[VAL3:.*]] = spirv.FAdd %[[VAL1]], %[[VAL2]]
      %16 = arith.addf %14, %15 : f32
      // CHECK: %[[PTR3:.*]] = spirv.AccessChain %[[ARG2]]{{\[}}{{%.*}}, {{%.*}}{{\]}}
      // CHECK-NEXT: spirv.Store "StorageBuffer" %[[PTR3]], %[[VAL3]]
      memref.store %16, %arg2[%12, %13] : memref<12x4xf32, #spirv.storage_class<StorageBuffer>>
      gpu.return
    }
  }
}
