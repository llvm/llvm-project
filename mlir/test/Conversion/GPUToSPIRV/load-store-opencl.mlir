// RUN: mlir-opt -convert-gpu-to-spirv='use-64bit-index=true use-opencl=true' %s -o - | FileCheck %s

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Kernel, Int64, Addresses], []>, api=OpenCL, #spirv.resource_limits<>>
} {
  func.func @load_store(%arg0: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg1: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg2: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>) {
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
        args(%arg0 : memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg1 : memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg2 : memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>,
             %c0 : index, %c0_0 : index, %c1 : index, %c1_1 : index)
    return
  }

  // CHECK-LABEL: spirv.module @{{.*}} Physical64 OpenCL
  gpu.module @kernels {
    // CHECK-DAG: spirv.GlobalVariable @[[WORKGROUPSIZEVAR:.*]] built_in("WorkgroupSize") : !spirv.ptr<vector<3xi64>, Input>
    // CHECK-DAG: spirv.GlobalVariable @[[NUMWORKGROUPSVAR:.*]] built_in("NumWorkgroups") : !spirv.ptr<vector<3xi64>, Input>
    // CHECK-DAG: spirv.GlobalVariable @[[$LOCALINVOCATIONIDVAR:.*]] built_in("LocalInvocationId") : !spirv.ptr<vector<3xi64>, Input>
    // CHECK-DAG: spirv.GlobalVariable @[[$WORKGROUPIDVAR:.*]] built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
    // CHECK-LABEL:    spirv.func @load_store_kernel
    // CHECK-SAME: %[[ARG0:[^\s]+]]: !spirv.ptr<!spirv.array<48 x f32>, CrossWorkgroup>
    // CHECK-SAME: %[[ARG1:[^\s]+]]: !spirv.ptr<!spirv.array<48 x f32>, CrossWorkgroup>
    // CHECK-SAME: %[[ARG2:[^\s]+]]: !spirv.ptr<!spirv.array<48 x f32>, CrossWorkgroup>
    gpu.func @load_store_kernel(%arg0: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg1: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg2: memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) kernel
      attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 16, 1, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
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
      // CHECK: %[[INDEX1:.*]] = spirv.IAdd
      %12 = arith.addi %arg3, %0 : index
      // CHECK: %[[INDEX2:.*]] = spirv.IAdd
      %13 = arith.addi %arg4, %3 : index
      // CHECK: %[[OFFSET1_0:.*]] = spirv.Constant 0 : i64
      // CHECK: %[[STRIDE1_1:.*]] = spirv.Constant 4 : i64
      // CHECK: %[[UPDATE1_1:.*]] = spirv.IMul %[[STRIDE1_1]], %[[INDEX1]] : i64
      // CHECK: %[[OFFSET1_1:.*]] = spirv.IAdd %[[OFFSET1_0]], %[[UPDATE1_1]] : i64
      // CHECK: %[[STRIDE1_2:.*]] = spirv.Constant 1 : i64
      // CHECK: %[[UPDATE1_2:.*]] = spirv.IMul %[[STRIDE1_2]], %[[INDEX2]] : i64
      // CHECK: %[[OFFSET1_2:.*]] = spirv.IAdd %[[OFFSET1_1]], %[[UPDATE1_2]] : i64
      // CHECK: %[[PTR1:.*]] = spirv.AccessChain %[[ARG0]]{{\[}}%[[OFFSET1_2]]{{\]}} : !spirv.ptr<!spirv.array<48 x f32>, CrossWorkgroup>, i64
      // CHECK-NEXT: %[[VAL1:.*]] = spirv.Load "CrossWorkgroup" %[[PTR1]] : f32
      %14 = memref.load %arg0[%12, %13] : memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>
      // CHECK: %[[PTR2:.*]] = spirv.AccessChain %[[ARG1]]{{\[}}{{%.*}}{{\]}}
      // CHECK-NEXT: %[[VAL2:.*]] = spirv.Load "CrossWorkgroup" %[[PTR2]]
      %15 = memref.load %arg1[%12, %13] : memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>
      // CHECK: %[[VAL3:.*]] = spirv.FAdd %[[VAL1]], %[[VAL2]] : f32
      %16 = arith.addf %14, %15 : f32
      // CHECK: %[[PTR3:.*]] = spirv.AccessChain %[[ARG2]]{{\[}}{{%.*}}{{\]}}
      // CHECK: spirv.Store "CrossWorkgroup" %[[PTR3]], %[[VAL3]] : f32
      memref.store %16, %arg2[%12, %13] : memref<12x4xf32, #spirv.storage_class<CrossWorkgroup>>
      // CHECK: spirv.Return
      gpu.return
    }
  }
}
