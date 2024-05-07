// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv))" -split-input-file -verify-diagnostics %s \
// RUN: | FileCheck --check-prefixes=CHECK-64,CHECK %s
// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv{index-bitwidth=32}))" -split-input-file -verify-diagnostics %s \
// RUN: | FileCheck --check-prefixes=CHECK-32,CHECK %s

gpu.module @builtins {
  // CHECK-64:    llvm.func spir_funccc @_Z14get_num_groupsj(i32) -> i64
  // CHECK-64:    llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64
  // CHECK-64:    llvm.func spir_funccc @_Z14get_local_sizej(i32) -> i64
  // CHECK-64:    llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64
  // CHECK-64:    llvm.func spir_funccc @_Z12get_group_idj(i32) -> i64
  // CHECK-32:    llvm.func spir_funccc @_Z14get_num_groupsj(i32) -> i32
  // CHECK-32:    llvm.func spir_funccc @_Z12get_local_idj(i32) -> i32
  // CHECK-32:    llvm.func spir_funccc @_Z14get_local_sizej(i32) -> i32
  // CHECK-32:    llvm.func spir_funccc @_Z13get_global_idj(i32) -> i32
  // CHECK-32:    llvm.func spir_funccc @_Z12get_group_idj(i32) -> i32

  // CHECK-LABEL: gpu_block_id
  func.func @gpu_block_id() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z12get_group_idj([[C0]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z12get_group_idj([[C0]]) : (i32) -> i32
    %block_id_x = gpu.block_id x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z12get_group_idj([[C1]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z12get_group_idj([[C1]]) : (i32) -> i32
    %block_id_y = gpu.block_id y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z12get_group_idj([[C2]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z12get_group_idj([[C2]]) : (i32) -> i32
    %block_id_z = gpu.block_id z
    return %block_id_x, %block_id_y, %block_id_z : index, index, index
  }

  // CHECK-LABEL: gpu_global_id
  func.func @gpu_global_id() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z13get_global_idj([[C0]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z13get_global_idj([[C0]]) : (i32) -> i32
    %global_id_x = gpu.global_id x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z13get_global_idj([[C1]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z13get_global_idj([[C1]]) : (i32) -> i32
    %global_id_y = gpu.global_id y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z13get_global_idj([[C2]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z13get_global_idj([[C2]]) : (i32) -> i32
    %global_id_z = gpu.global_id z
    return %global_id_x, %global_id_y, %global_id_z : index, index, index
  }

  // CHECK-LABEL: gpu_block_dim
  func.func @gpu_block_dim() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z14get_local_sizej([[C0]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z14get_local_sizej([[C0]]) : (i32) -> i32
    %block_dim_x = gpu.block_dim x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z14get_local_sizej([[C1]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z14get_local_sizej([[C1]]) : (i32) -> i32
    %block_dim_y = gpu.block_dim y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z14get_local_sizej([[C2]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z14get_local_sizej([[C2]]) : (i32) -> i32
    %block_dim_z = gpu.block_dim z
    return %block_dim_x, %block_dim_y, %block_dim_z : index, index, index
  }

  // CHECK-LABEL: gpu_thread_id
  func.func @gpu_thread_id() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z12get_local_idj([[C0]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z12get_local_idj([[C0]]) : (i32) -> i32
    %thread_id_x = gpu.thread_id x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z12get_local_idj([[C1]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z12get_local_idj([[C1]]) : (i32) -> i32
    %thread_id_y = gpu.thread_id y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z12get_local_idj([[C2]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z12get_local_idj([[C2]]) : (i32) -> i32
    %thread_id_z = gpu.thread_id z
    return %thread_id_x, %thread_id_y, %thread_id_z : index, index, index
  }

  // CHECK-LABEL: gpu_grid_dim
  func.func @gpu_grid_dim() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z14get_num_groupsj([[C0]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z14get_num_groupsj([[C0]]) : (i32) -> i32
    %grid_dim_x = gpu.grid_dim x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z14get_num_groupsj([[C1]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z14get_num_groupsj([[C1]]) : (i32) -> i32
    %grid_dim_y = gpu.grid_dim y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-64:      llvm.call spir_funccc @_Z14get_num_groupsj([[C2]]) : (i32) -> i64
    // CHECK-32:      llvm.call spir_funccc @_Z14get_num_groupsj([[C2]]) : (i32) -> i32
    %grid_dim_z = gpu.grid_dim z
    return %grid_dim_x, %grid_dim_y, %grid_dim_z : index, index, index
  }
}

// -----

gpu.module @barriers {
  // CHECK:       llvm.func spir_funccc @_Z7barrierj(i32)

  // CHECK-LABEL: gpu_barrier
  func.func @gpu_barrier() {
    // CHECK:         [[FLAGS:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z7barrierj([[FLAGS]]) : (i32) -> ()
    gpu.barrier
    return
  }
}

// -----

// Check `gpu.shuffle` conversion with default subgroup size.

gpu.module @shuffles {
  // CHECK:       llvm.func spir_funccc @_Z22sub_group_shuffle_downdj(f64, i32) -> f64
  // CHECK:       llvm.func spir_funccc @_Z20sub_group_shuffle_upfj(f32, i32) -> f32
  // CHECK:       llvm.func spir_funccc @_Z21sub_group_shuffle_xorlj(i64, i32) -> i64
  // CHECK:       llvm.func spir_funccc @_Z17sub_group_shuffleij(i32, i32) -> i32

  // CHECK-LABEL: gpu_shuffles
  // CHECK-SAME:              (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: i32)
  func.func @gpu_shuffles(%val0: i32, %id: i32,
                          %val1: i64, %mask: i32,
                          %val2: f32, %delta_up: i32,
                          %val3: f64, %delta_down: i32) {
    %width = arith.constant 32 : i32
    // CHECK:         llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z21sub_group_shuffle_xorlj(%[[VAL_2]], %[[VAL_3]]) : (i64, i32) -> i64
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z20sub_group_shuffle_upfj(%[[VAL_4]], %[[VAL_5]]) : (f32, i32) -> f32
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z22sub_group_shuffle_downdj(%[[VAL_6]], %[[VAL_7]]) : (f64, i32) -> f64
    // CHECK:         llvm.mlir.constant(true) : i1
    %shuffleResult0, %valid0 = gpu.shuffle idx %val0, %id, %width : i32
    %shuffleResult1, %valid1 = gpu.shuffle xor %val1, %mask, %width : i64
    %shuffleResult2, %valid2 = gpu.shuffle up %val2, %delta_up, %width : f32
    %shuffleResult3, %valid3 = gpu.shuffle down %val3, %delta_down, %width : f64
    return
  }
}

// -----

// Check `gpu.shuffle` conversion with explicit subgroup size.

gpu.module @shuffles attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {
  // CHECK:       llvm.func spir_funccc @_Z22sub_group_shuffle_downdj(f64, i32) -> f64
  // CHECK:       llvm.func spir_funccc @_Z20sub_group_shuffle_upfj(f32, i32) -> f32
  // CHECK:       llvm.func spir_funccc @_Z21sub_group_shuffle_xorlj(i64, i32) -> i64
  // CHECK:       llvm.func spir_funccc @_Z17sub_group_shuffleij(i32, i32) -> i32

  // CHECK-LABEL: gpu_shuffles
  // CHECK-SAME:              (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: i32)
  func.func @gpu_shuffles(%val0: i32, %id: i32,
                          %val1: i64, %mask: i32,
                          %val2: f32, %delta_up: i32,
                          %val3: f64, %delta_down: i32) {
    %width = arith.constant 16 : i32
    // CHECK:         llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[VAL_0]], %[[VAL_1]]) : (i32, i32) -> i32
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z21sub_group_shuffle_xorlj(%[[VAL_2]], %[[VAL_3]]) : (i64, i32) -> i64
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z20sub_group_shuffle_upfj(%[[VAL_4]], %[[VAL_5]]) : (f32, i32) -> f32
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z22sub_group_shuffle_downdj(%[[VAL_6]], %[[VAL_7]]) : (f64, i32) -> f64
    // CHECK:         llvm.mlir.constant(true) : i1
    %shuffleResult0, %valid0 = gpu.shuffle idx %val0, %id, %width : i32
    %shuffleResult1, %valid1 = gpu.shuffle xor %val1, %mask, %width : i64
    %shuffleResult2, %valid2 = gpu.shuffle up %val2, %delta_up, %width : f32
    %shuffleResult3, %valid3 = gpu.shuffle down %val3, %delta_down, %width : f64
    return
  }
}

// -----

// Cannot convert due to shuffle width and target subgroup size mismatch

gpu.module @shuffles_mismatch {
  func.func @gpu_shuffles(%val: i32, %id: i32) {
    %width = arith.constant 16 : i32
    // expected-error@below {{failed to legalize operation 'gpu.shuffle' that was explicitly marked illegal}}
    %shuffleResult, %valid = gpu.shuffle idx %val, %id, %width : i32
    return
  }
}

// -----

// Cannot convert due to variable shuffle width

gpu.module @shuffles_mismatch {
  func.func @gpu_shuffles(%val: i32, %id: i32, %width: i32) {
    // expected-error@below {{failed to legalize operation 'gpu.shuffle' that was explicitly marked illegal}}
    %shuffleResult, %valid = gpu.shuffle idx %val, %id, %width : i32
    return
  }
}
