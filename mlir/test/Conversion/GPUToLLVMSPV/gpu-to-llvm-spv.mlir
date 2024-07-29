// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv))" -split-input-file -verify-diagnostics %s \
// RUN: | FileCheck --check-prefixes=CHECK-64,CHECK %s
// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv{index-bitwidth=32}))" -split-input-file -verify-diagnostics %s \
// RUN: | FileCheck --check-prefixes=CHECK-32,CHECK %s

gpu.module @builtins {
  // CHECK-64:        llvm.func spir_funccc @_Z14get_num_groupsj(i32) -> i64 attributes {
  // CHECK-32:        llvm.func spir_funccc @_Z14get_num_groupsj(i32) -> i32 attributes {
  // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       convergent
  // CHECK-SAME:      }
  // CHECK-64:        llvm.func spir_funccc @_Z12get_local_idj(i32) -> i64 attributes {
  // CHECK-32:        llvm.func spir_funccc @_Z12get_local_idj(i32) -> i32 attributes {
  // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       convergent
  // CHECK-SAME:      }
  // CHECK-64:        llvm.func spir_funccc @_Z14get_local_sizej(i32) -> i64 attributes {
  // CHECK-32:        llvm.func spir_funccc @_Z14get_local_sizej(i32) -> i32 attributes {
  // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       convergent
  // CHECK-SAME:      }
  // CHECK-64:        llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64 attributes {
  // CHECK-32:        llvm.func spir_funccc @_Z13get_global_idj(i32) -> i32 attributes {
  // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       convergent
  // CHECK-SAME:      }
  // CHECK-64:        llvm.func spir_funccc @_Z12get_group_idj(i32) -> i64 attributes {
  // CHECK-32:        llvm.func spir_funccc @_Z12get_group_idj(i32) -> i32 attributes {
  // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       convergent
  // CHECK-SAME:      }

  // CHECK-LABEL: gpu_block_id
  func.func @gpu_block_id() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z12get_group_idj([[C0]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %block_id_x = gpu.block_id x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z12get_group_idj([[C1]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %block_id_y = gpu.block_id y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z12get_group_idj([[C2]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %block_id_z = gpu.block_id z
    return %block_id_x, %block_id_y, %block_id_z : index, index, index
  }

  // CHECK-LABEL: gpu_global_id
  func.func @gpu_global_id() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z13get_global_idj([[C0]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %global_id_x = gpu.global_id x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z13get_global_idj([[C1]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %global_id_y = gpu.global_id y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z13get_global_idj([[C2]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %global_id_z = gpu.global_id z
    return %global_id_x, %global_id_y, %global_id_z : index, index, index
  }

  // CHECK-LABEL: gpu_block_dim
  func.func @gpu_block_dim() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z14get_local_sizej([[C0]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %block_dim_x = gpu.block_dim x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z14get_local_sizej([[C1]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %block_dim_y = gpu.block_dim y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z14get_local_sizej([[C2]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %block_dim_z = gpu.block_dim z
    return %block_dim_x, %block_dim_y, %block_dim_z : index, index, index
  }

  // CHECK-LABEL: gpu_thread_id
  func.func @gpu_thread_id() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z12get_local_idj([[C0]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %thread_id_x = gpu.thread_id x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z12get_local_idj([[C1]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %thread_id_y = gpu.thread_id y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z12get_local_idj([[C2]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %thread_id_z = gpu.thread_id z
    return %thread_id_x, %thread_id_y, %thread_id_z : index, index, index
  }

  // CHECK-LABEL: gpu_grid_dim
  func.func @gpu_grid_dim() -> (index, index, index) {
    // CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z14get_num_groupsj([[C0]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %grid_dim_x = gpu.grid_dim x
    // CHECK:         [[C1:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z14get_num_groupsj([[C1]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %grid_dim_y = gpu.grid_dim y
    // CHECK:         [[C2:%.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z14get_num_groupsj([[C2]]) {
    // CHECK-SAME-DAG:  memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       convergent
    // CHECK-64-SAME: } : (i32) -> i64
    // CHECK-32-SAME: } : (i32) -> i32
    %grid_dim_z = gpu.grid_dim z
    return %grid_dim_x, %grid_dim_y, %grid_dim_z : index, index, index
  }
}

// -----

gpu.module @barriers {
  // CHECK:           llvm.func spir_funccc @_Z7barrierj(i32) attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }

  // CHECK-LABEL: gpu_barrier
  func.func @gpu_barrier() {
    // CHECK:         [[FLAGS:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK:         llvm.call spir_funccc @_Z7barrierj([[FLAGS]]) {
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  convergent
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       memory_effects = #llvm.memory_effects
    // CHECK-SAME:    } : (i32) -> ()
    gpu.barrier
    return
  }
}

// -----

// Check `gpu.shuffle` conversion with default subgroup size.

gpu.module @shuffles {
  // CHECK:           llvm.func spir_funccc @_Z22sub_group_shuffle_downdj(f64, i32) -> f64 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }
  // CHECK:           llvm.func spir_funccc @_Z20sub_group_shuffle_upfj(f32, i32) -> f32 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }
  // CHECK:           llvm.func spir_funccc @_Z21sub_group_shuffle_xorlj(i64, i32) -> i64 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }
  // CHECK:           llvm.func spir_funccc @_Z17sub_group_shuffleij(i32, i32) -> i32 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }

  // CHECK-LABEL: gpu_shuffles
  // CHECK-SAME:              (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: i32)
  func.func @gpu_shuffles(%val0: i32, %id: i32,
                          %val1: i64, %mask: i32,
                          %val2: f32, %delta_up: i32,
                          %val3: f64, %delta_down: i32) {
    %width = arith.constant 32 : i32
    // CHECK:         llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[VAL_0]], %[[VAL_1]]) {
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  convergent
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       memory_effects = #llvm.memory_effects
    // CHECK-SAME:    } : (i32, i32) -> i32
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z21sub_group_shuffle_xorlj(%[[VAL_2]], %[[VAL_3]]) {
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  convergent
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       memory_effects = #llvm.memory_effects
    // CHECK-SAME:    } : (i64, i32) -> i64
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z20sub_group_shuffle_upfj(%[[VAL_4]], %[[VAL_5]]) {
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  convergent
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       memory_effects= #llvm.memory_effects
    // CHECK-SAME:    } : (f32, i32) -> f32
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z22sub_group_shuffle_downdj(%[[VAL_6]], %[[VAL_7]]) {
    // CHECK-SAME-DAG:  no_unwind
    // CHECK-SAME-DAG:  convergent
    // CHECK-SAME-DAG:  will_return
    // CHECK-NOT:       memory_effects= #llvm.memory_effects
    // CHECK-SAME:    } : (f64, i32) -> f64
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
  // CHECK:           llvm.func spir_funccc @_Z22sub_group_shuffle_downdj(f64, i32) -> f64 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }
  // CHECK:           llvm.func spir_funccc @_Z20sub_group_shuffle_upfj(f32, i32) -> f32 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }
  // CHECK:           llvm.func spir_funccc @_Z21sub_group_shuffle_xorlj(i64, i32) -> i64 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }
  // CHECK:           llvm.func spir_funccc @_Z17sub_group_shuffleij(i32, i32) -> i32 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }

  // CHECK-LABEL: gpu_shuffles
  // CHECK-SAME:              (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: f64, %[[VAL_7:.*]]: i32)
  func.func @gpu_shuffles(%val0: i32, %id: i32,
                          %val1: i64, %mask: i32,
                          %val2: f32, %delta_up: i32,
                          %val3: f64, %delta_down: i32) {
    %width = arith.constant 16 : i32
    // CHECK:         llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[VAL_0]], %[[VAL_1]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z21sub_group_shuffle_xorlj(%[[VAL_2]], %[[VAL_3]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z20sub_group_shuffle_upfj(%[[VAL_4]], %[[VAL_5]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z22sub_group_shuffle_downdj(%[[VAL_6]], %[[VAL_7]])
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

// -----

gpu.module @kernels {
// CHECK:           llvm.func spir_funccc @no_kernel() {
  gpu.func @no_kernel() {
    gpu.return
  }

// CHECK:           llvm.func spir_kernelcc @kernel_no_arg() attributes {gpu.kernel} {
  gpu.func @kernel_no_arg() kernel {
    gpu.return
  }

// CHECK:           llvm.func spir_kernelcc @kernel_with_args(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: i64) attributes {gpu.kernel} {
  gpu.func @kernel_with_args(%arg0: f32, %arg1: i64) kernel {
    gpu.return
  }

// CHECK-64:           llvm.func spir_kernelcc @kernel_with_conv_args(%[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: !llvm.ptr, %[[VAL_4:.*]]: !llvm.ptr, %[[VAL_5:.*]]: i64) attributes {gpu.kernel} {
// CHECK-32:           llvm.func spir_kernelcc @kernel_with_conv_args(%[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !llvm.ptr, %[[VAL_4:.*]]: !llvm.ptr, %[[VAL_5:.*]]: i32) attributes {gpu.kernel} {
  gpu.func @kernel_with_conv_args(%arg0: index, %arg1: memref<index>) kernel {
    gpu.return
  }

// CHECK-64:           llvm.func spir_kernelcc @kernel_with_sized_memref(%[[VAL_6:.*]]: !llvm.ptr, %[[VAL_7:.*]]: !llvm.ptr, %[[VAL_8:.*]]: i64, %[[VAL_9:.*]]: i64, %[[VAL_10:.*]]: i64) attributes {gpu.kernel} {
// CHECK-32:           llvm.func spir_kernelcc @kernel_with_sized_memref(%[[VAL_6:.*]]: !llvm.ptr, %[[VAL_7:.*]]: !llvm.ptr, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32) attributes {gpu.kernel} {
  gpu.func @kernel_with_sized_memref(%arg0: memref<1xindex>) kernel {
    gpu.return
  }

// CHECK-64:           llvm.func spir_kernelcc @kernel_with_ND_memref(%[[VAL_11:.*]]: !llvm.ptr, %[[VAL_12:.*]]: !llvm.ptr, %[[VAL_13:.*]]: i64, %[[VAL_14:.*]]: i64, %[[VAL_15:.*]]: i64, %[[VAL_16:.*]]: i64, %[[VAL_17:.*]]: i64, %[[VAL_18:.*]]: i64, %[[VAL_19:.*]]: i64) attributes {gpu.kernel} {
// CHECK-32:           llvm.func spir_kernelcc @kernel_with_ND_memref(%[[VAL_11:.*]]: !llvm.ptr, %[[VAL_12:.*]]: !llvm.ptr, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i32) attributes {gpu.kernel} {
  gpu.func @kernel_with_ND_memref(%arg0: memref<128x128x128xindex>) kernel {
    gpu.return
  }
}

// -----

gpu.module @kernels {
// CHECK-LABEL:           llvm.func spir_kernelcc @kernel_with_private_attribs(
// CHECK-SAME:                %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: i16) attributes {gpu.kernel} {
// CHECK:                   %[[VAL_2:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:                   %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x f32 : (i64) -> !llvm.ptr

// CHECK-64:             %[[VAL_4:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_4]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_5]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_6]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_9:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK-64:             %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_8]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_11:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_10]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_13:.*]] = builtin.unrealized_conversion_cast %[[VAL_12]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<32xf32>

// CHECK-32:             %[[VAL_4:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_4]][0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_6:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_5]][1] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_7:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_8:.*]] = llvm.insertvalue %[[VAL_7]], %[[VAL_6]][2] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_9:.*]] = llvm.mlir.constant(32 : index) : i32
// CHECK-32:             %[[VAL_10:.*]] = llvm.insertvalue %[[VAL_9]], %[[VAL_8]][3, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_11:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_12:.*]] = llvm.insertvalue %[[VAL_11]], %[[VAL_10]][4, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_13:.*]] = builtin.unrealized_conversion_cast %[[VAL_12]] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)> to memref<32xf32>

// CHECK:                %[[VAL_14:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:                %[[VAL_15:.*]] = llvm.alloca %[[VAL_14]] x i16 : (i64) -> !llvm.ptr

// CHECK-64:             %[[VAL_16:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_16]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_17]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_19:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_20:.*]] = llvm.insertvalue %[[VAL_19]], %[[VAL_18]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_21:.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK-64:             %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_21]], %[[VAL_20]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_23:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_22]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_25:.*]] = builtin.unrealized_conversion_cast %[[VAL_24]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<16xi16>

// CHECK-32:             %[[VAL_16:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_16]][0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_17]][1] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_19:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_20:.*]] = llvm.insertvalue %[[VAL_19]], %[[VAL_18]][2] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_21:.*]] = llvm.mlir.constant(16 : index) : i32
// CHECK-32:             %[[VAL_22:.*]] = llvm.insertvalue %[[VAL_21]], %[[VAL_20]][3, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_23:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_24:.*]] = llvm.insertvalue %[[VAL_23]], %[[VAL_22]][4, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_25:.*]] = builtin.unrealized_conversion_cast %[[VAL_24]] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)> to memref<16xi16>

// CHECK:                %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:                memref.store %[[VAL_0]], %[[VAL_13]]{{\[}}%[[VAL_26]]] : memref<32xf32>
// CHECK:                memref.store %[[VAL_1]], %[[VAL_25]]{{\[}}%[[VAL_26]]] : memref<16xi16>
  gpu.func @kernel_with_private_attribs(%arg0: f32, %arg1: i16)
      private(%arg2: memref<32xf32>, %arg3: memref<16xi16>)
      kernel {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg2[%c0] : memref<32xf32>
    memref.store %arg1, %arg3[%c0] : memref<16xi16>
    gpu.return
  }

// CHECK-LABEL:        llvm.func spir_kernelcc @kernel_with_workgoup_attribs(
// CHECK-SAME:             %[[VAL_27:.*]]: f32, %[[VAL_28:.*]]: i16, %[[VAL_29:.*]]: !llvm.ptr<3>, %[[VAL_30:.*]]: !llvm.ptr<3>) attributes {gpu.kernel} {

// CHECK-64:             %[[VAL_31:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_31]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_32]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_34:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_34]], %[[VAL_33]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_36:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK-64:             %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_35]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_38:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_39:.*]] = llvm.insertvalue %[[VAL_38]], %[[VAL_37]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_40:.*]] = builtin.unrealized_conversion_cast %[[VAL_39]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> to memref<32xf32, 3>
// CHECK-64:             %[[VAL_41:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_41]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_42]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_44:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_45:.*]] = llvm.insertvalue %[[VAL_44]], %[[VAL_43]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_46:.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK-64:             %[[VAL_47:.*]] = llvm.insertvalue %[[VAL_46]], %[[VAL_45]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_48:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_49:.*]] = llvm.insertvalue %[[VAL_48]], %[[VAL_47]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_50:.*]] = builtin.unrealized_conversion_cast %[[VAL_49]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> to memref<16xi16, 3>

// CHECK-32:             %[[VAL_31:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_31]][0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_33:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_32]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_34:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_35:.*]] = llvm.insertvalue %[[VAL_34]], %[[VAL_33]][2] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_36:.*]] = llvm.mlir.constant(32 : index) : i32
// CHECK-32:             %[[VAL_37:.*]] = llvm.insertvalue %[[VAL_36]], %[[VAL_35]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_38:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_39:.*]] = llvm.insertvalue %[[VAL_38]], %[[VAL_37]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_40:.*]] = builtin.unrealized_conversion_cast %[[VAL_39]] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)> to memref<32xf32, 3>
// CHECK-32:             %[[VAL_41:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_41]][0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_42]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_44:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_45:.*]] = llvm.insertvalue %[[VAL_44]], %[[VAL_43]][2] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_46:.*]] = llvm.mlir.constant(16 : index) : i32
// CHECK-32:             %[[VAL_47:.*]] = llvm.insertvalue %[[VAL_46]], %[[VAL_45]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_48:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_49:.*]] = llvm.insertvalue %[[VAL_48]], %[[VAL_47]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_50:.*]] = builtin.unrealized_conversion_cast %[[VAL_49]] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)> to memref<16xi16, 3>

// CHECK:                %[[VAL_51:.*]] = arith.constant 0 : index
// CHECK:                memref.store %[[VAL_27]], %[[VAL_40]]{{\[}}%[[VAL_51]]] : memref<32xf32, 3>
// CHECK:                memref.store %[[VAL_28]], %[[VAL_50]]{{\[}}%[[VAL_51]]] : memref<16xi16, 3>
  gpu.func @kernel_with_workgoup_attribs(%arg0: f32, %arg1: i16)
      workgroup(%arg2: memref<32xf32, 3>, %arg3: memref<16xi16, 3>)
      kernel {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg2[%c0] : memref<32xf32, 3>
    memref.store %arg1, %arg3[%c0] : memref<16xi16, 3>
    gpu.return
  }

// CHECK-LABEL:        llvm.func spir_kernelcc @kernel_with_both_attribs(
// CHECK-64-SAME:          %[[VAL_52:.*]]: f32, %[[VAL_53:.*]]: i16, %[[VAL_54:.*]]: i32, %[[VAL_55:.*]]: i64, %[[VAL_56:.*]]: !llvm.ptr<3>, %[[VAL_57:.*]]: !llvm.ptr<3>) attributes {gpu.kernel} {
// CHECK-32-SAME:          %[[VAL_52:.*]]: f32, %[[VAL_53:.*]]: i16, %[[VAL_54:.*]]: i32, %[[VAL_55:.*]]: i32, %[[VAL_56:.*]]: !llvm.ptr<3>, %[[VAL_57:.*]]: !llvm.ptr<3>) attributes {gpu.kernel} {

// CHECK-64:             %[[VAL_58:.*]] = builtin.unrealized_conversion_cast %[[VAL_55]] : i64 to index
// CHECK-64:             %[[VAL_59:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_60:.*]] = llvm.insertvalue %[[VAL_56]], %[[VAL_59]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_61:.*]] = llvm.insertvalue %[[VAL_56]], %[[VAL_60]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_62:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_63:.*]] = llvm.insertvalue %[[VAL_62]], %[[VAL_61]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_64:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK-64:             %[[VAL_65:.*]] = llvm.insertvalue %[[VAL_64]], %[[VAL_63]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_66:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_67:.*]] = llvm.insertvalue %[[VAL_66]], %[[VAL_65]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_68:.*]] = builtin.unrealized_conversion_cast %[[VAL_67]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> to memref<32xf32, 3>
// CHECK-64:             %[[VAL_69:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_70:.*]] = llvm.insertvalue %[[VAL_57]], %[[VAL_69]][0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_71:.*]] = llvm.insertvalue %[[VAL_57]], %[[VAL_70]][1] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_72:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_73:.*]] = llvm.insertvalue %[[VAL_72]], %[[VAL_71]][2] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_74:.*]] = llvm.mlir.constant(16 : index) : i64
// CHECK-64:             %[[VAL_75:.*]] = llvm.insertvalue %[[VAL_74]], %[[VAL_73]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_76:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_77:.*]] = llvm.insertvalue %[[VAL_76]], %[[VAL_75]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_78:.*]] = builtin.unrealized_conversion_cast %[[VAL_77]] : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)> to memref<16xi16, 3>

// CHECK-32:             %[[VAL_58:.*]] = builtin.unrealized_conversion_cast %[[VAL_55]] : i32 to index
// CHECK-32:             %[[VAL_59:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_60:.*]] = llvm.insertvalue %[[VAL_56]], %[[VAL_59]][0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_61:.*]] = llvm.insertvalue %[[VAL_56]], %[[VAL_60]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_62:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_63:.*]] = llvm.insertvalue %[[VAL_62]], %[[VAL_61]][2] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_64:.*]] = llvm.mlir.constant(32 : index) : i32
// CHECK-32:             %[[VAL_65:.*]] = llvm.insertvalue %[[VAL_64]], %[[VAL_63]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_66:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_67:.*]] = llvm.insertvalue %[[VAL_66]], %[[VAL_65]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_68:.*]] = builtin.unrealized_conversion_cast %[[VAL_67]] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)> to memref<32xf32, 3>
// CHECK-32:             %[[VAL_69:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_70:.*]] = llvm.insertvalue %[[VAL_57]], %[[VAL_69]][0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_71:.*]] = llvm.insertvalue %[[VAL_57]], %[[VAL_70]][1] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_72:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_73:.*]] = llvm.insertvalue %[[VAL_72]], %[[VAL_71]][2] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_74:.*]] = llvm.mlir.constant(16 : index) : i32
// CHECK-32:             %[[VAL_75:.*]] = llvm.insertvalue %[[VAL_74]], %[[VAL_73]][3, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_76:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_77:.*]] = llvm.insertvalue %[[VAL_76]], %[[VAL_75]][4, 0] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_78:.*]] = builtin.unrealized_conversion_cast %[[VAL_77]] : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)> to memref<16xi16, 3>

// CHECK:                %[[VAL_79:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:                %[[VAL_80:.*]] = llvm.alloca %[[VAL_79]] x i32 : (i64) -> !llvm.ptr

// CHECK-64:             %[[VAL_81:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_82:.*]] = llvm.insertvalue %[[VAL_80]], %[[VAL_81]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_83:.*]] = llvm.insertvalue %[[VAL_80]], %[[VAL_82]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_84:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_85:.*]] = llvm.insertvalue %[[VAL_84]], %[[VAL_83]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_86:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK-64:             %[[VAL_87:.*]] = llvm.insertvalue %[[VAL_86]], %[[VAL_85]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_88:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_89:.*]] = llvm.insertvalue %[[VAL_88]], %[[VAL_87]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_90:.*]] = builtin.unrealized_conversion_cast %[[VAL_89]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<32xi32>

// CHECK-32:             %[[VAL_81:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_82:.*]] = llvm.insertvalue %[[VAL_80]], %[[VAL_81]][0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_83:.*]] = llvm.insertvalue %[[VAL_80]], %[[VAL_82]][1] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_84:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_85:.*]] = llvm.insertvalue %[[VAL_84]], %[[VAL_83]][2] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_86:.*]] = llvm.mlir.constant(32 : index) : i32
// CHECK-32:             %[[VAL_87:.*]] = llvm.insertvalue %[[VAL_86]], %[[VAL_85]][3, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_88:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_89:.*]] = llvm.insertvalue %[[VAL_88]], %[[VAL_87]][4, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_90:.*]] = builtin.unrealized_conversion_cast %[[VAL_89]] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)> to memref<32xi32>

// CHECK:                %[[VAL_91:.*]] = llvm.mlir.constant(32 : i64) : i64

// CHECK-64:             %[[VAL_92:.*]] = llvm.alloca %[[VAL_91]] x i64 : (i64) -> !llvm.ptr
// CHECK-32:             %[[VAL_92:.*]] = llvm.alloca %[[VAL_91]] x i32 : (i64) -> !llvm.ptr

// CHECK-64:             %[[VAL_93:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_94:.*]] = llvm.insertvalue %[[VAL_92]], %[[VAL_93]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_95:.*]] = llvm.insertvalue %[[VAL_92]], %[[VAL_94]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_96:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-64:             %[[VAL_97:.*]] = llvm.insertvalue %[[VAL_96]], %[[VAL_95]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_98:.*]] = llvm.mlir.constant(32 : index) : i64
// CHECK-64:             %[[VAL_99:.*]] = llvm.insertvalue %[[VAL_98]], %[[VAL_97]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_100:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-64:             %[[VAL_101:.*]] = llvm.insertvalue %[[VAL_100]], %[[VAL_99]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-64:             %[[VAL_102:.*]] = builtin.unrealized_conversion_cast %[[VAL_101]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<32xindex>

// CHECK-32:             %[[VAL_93:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_94:.*]] = llvm.insertvalue %[[VAL_92]], %[[VAL_93]][0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_95:.*]] = llvm.insertvalue %[[VAL_92]], %[[VAL_94]][1] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_96:.*]] = llvm.mlir.constant(0 : index) : i32
// CHECK-32:             %[[VAL_97:.*]] = llvm.insertvalue %[[VAL_96]], %[[VAL_95]][2] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_98:.*]] = llvm.mlir.constant(32 : index) : i32
// CHECK-32:             %[[VAL_99:.*]] = llvm.insertvalue %[[VAL_98]], %[[VAL_97]][3, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_100:.*]] = llvm.mlir.constant(1 : index) : i32
// CHECK-32:             %[[VAL_101:.*]] = llvm.insertvalue %[[VAL_100]], %[[VAL_99]][4, 0] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>
// CHECK-32:             %[[VAL_102:.*]] = builtin.unrealized_conversion_cast %[[VAL_101]] : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)> to memref<32xindex>

// CHECK:                %[[VAL_103:.*]] = arith.constant 0 : index
// CHECK:                memref.store %[[VAL_52]], %[[VAL_68]]{{\[}}%[[VAL_103]]] : memref<32xf32, 3>
// CHECK:                memref.store %[[VAL_53]], %[[VAL_78]]{{\[}}%[[VAL_103]]] : memref<16xi16, 3>
// CHECK:                memref.store %[[VAL_54]], %[[VAL_90]]{{\[}}%[[VAL_103]]] : memref<32xi32>
// CHECK:                memref.store %[[VAL_58]], %[[VAL_102]]{{\[}}%[[VAL_103]]] : memref<32xindex>
  gpu.func @kernel_with_both_attribs(%arg0: f32, %arg1: i16, %arg2: i32, %arg3: index)
      workgroup(%arg4: memref<32xf32, 3>, %arg5: memref<16xi16, 3>)
      private(%arg6: memref<32xi32>, %arg7: memref<32xindex>)
      kernel {
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg4[%c0] : memref<32xf32, 3>
    memref.store %arg1, %arg5[%c0] : memref<16xi16, 3>
    memref.store %arg2, %arg6[%c0] : memref<32xi32>
    memref.store %arg3, %arg7[%c0] : memref<32xindex>
    gpu.return
  }

// CHECK-LABEL:     llvm.func spir_kernelcc @kernel_known_block_size
// CHECK-SAME:          reqd_work_group_size = array<i32: 128, 128, 256>
  gpu.func @kernel_known_block_size() kernel attributes {known_block_size = array<i32: 128, 128, 256>} {
    gpu.return
  }
}
