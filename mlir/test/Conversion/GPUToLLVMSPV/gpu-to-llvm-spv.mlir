// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv{use-64bit-index=true}))" -split-input-file -verify-diagnostics %s \
// RUN: | FileCheck --check-prefixes=CHECK-64,CHECK %s
// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv))" -split-input-file -verify-diagnostics %s \
// RUN: | FileCheck --check-prefixes=CHECK-32,CHECK %s
// RUN: mlir-opt -pass-pipeline="builtin.module(gpu.module(convert-gpu-to-llvm-spv{use-64bit-index=false}))" -split-input-file -verify-diagnostics %s \
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

// Check `gpu.shuffle` conversion with explicit subgroup size.

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
  // CHECK:           llvm.func spir_funccc @_Z20sub_group_shuffle_upDhj(f16, i32) -> f16 attributes {
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
  // CHECK:           llvm.func spir_funccc @_Z21sub_group_shuffle_xorsj(i16, i32) -> i16 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }
  // CHECK:           llvm.func spir_funccc @_Z17sub_group_shufflecj(i8, i32) -> i8 attributes {
  // CHECK-SAME-DAG:  no_unwind
  // CHECK-SAME-DAG:  convergent
  // CHECK-SAME-DAG:  will_return
  // CHECK-NOT:       memory_effects = #llvm.memory_effects
  // CHECK-SAME:      }

  // CHECK-LABEL: gpu_shuffles
  // CHECK-SAME:              (%[[I8_VAL:.*]]: i8, %[[I16_VAL:.*]]: i16,
  // CHECK-SAME:               %[[I32_VAL:.*]]: i32, %[[I64_VAL:.*]]: i64,
  // CHECK-SAME:               %[[F16_VAL:.*]]: f16, %[[F32_VAL:.*]]: f32,
  // CHECK-SAME:               %[[F64_VAL:.*]]: f64, %[[BF16_VAL:.*]]: bf16,
  // CHECK-SAME:               %[[I1_VAL:.*]]: i1, %[[OFFSET:.*]]: i32)
  llvm.func @gpu_shuffles(%i8_val: i8,
                          %i16_val: i16,
                          %i32_val: i32,
                          %i64_val: i64,
                          %f16_val: f16,
                          %f32_val: f32,
                          %f64_val: f64,
                          %bf16_val: bf16,
                          %i1_val: i1,
                          %offset: i32) attributes {intel_reqd_sub_group_size = 16 : i32} {
    %width = arith.constant 16 : i32
    // CHECK:         llvm.call spir_funccc @_Z17sub_group_shufflecj(%[[I8_VAL]], %[[OFFSET]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z21sub_group_shuffle_xorsj(%[[I16_VAL]], %[[OFFSET]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z17sub_group_shuffleij(%[[I32_VAL]], %[[OFFSET]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z21sub_group_shuffle_xorlj(%[[I64_VAL]], %[[OFFSET]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z20sub_group_shuffle_upDhj(%[[F16_VAL]], %[[OFFSET]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z20sub_group_shuffle_upfj(%[[F32_VAL]], %[[OFFSET]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         llvm.call spir_funccc @_Z22sub_group_shuffle_downdj(%[[F64_VAL]], %[[OFFSET]])
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         %[[BF16_INBC:.*]] = llvm.bitcast %[[BF16_VAL]] : bf16 to i16
    // CHECK:         %[[BF16_CALL:.*]] = llvm.call spir_funccc @_Z22sub_group_shuffle_downsj(%[[BF16_INBC]], %[[OFFSET]])
    // CHECK:         llvm.bitcast %[[BF16_CALL]] : i16 to bf16
    // CHECK:         llvm.mlir.constant(true) : i1
    // CHECK:         %[[I1_ZEXT:.*]] = llvm.zext %[[I1_VAL]] : i1 to i8
    // CHECK:         %[[I1_CALL:.*]] = llvm.call spir_funccc @_Z21sub_group_shuffle_xorcj(%18, %arg9)
    // CHECK:         llvm.trunc %[[I1_CALL:.*]] : i8 to i1
    // CHECK:         llvm.mlir.constant(true) : i1
    %shuffleResult0, %valid0 = gpu.shuffle idx %i8_val, %offset, %width : i8
    %shuffleResult1, %valid1 = gpu.shuffle xor %i16_val, %offset, %width : i16
    %shuffleResult2, %valid2 = gpu.shuffle idx %i32_val, %offset, %width : i32
    %shuffleResult3, %valid3 = gpu.shuffle xor %i64_val, %offset, %width : i64
    %shuffleResult4, %valid4 = gpu.shuffle up %f16_val, %offset, %width : f16
    %shuffleResult5, %valid5 = gpu.shuffle up %f32_val, %offset, %width : f32
    %shuffleResult6, %valid6 = gpu.shuffle down %f64_val, %offset, %width : f64
    %shuffleResult7, %valid7 = gpu.shuffle down %bf16_val, %offset, %width : bf16
    %shuffleResult8, %valid8 = gpu.shuffle xor %i1_val, %offset, %width : i1
    llvm.return
  }
}

// -----

// Cannot convert due to shuffle width and target subgroup size mismatch

gpu.module @shuffles_mismatch {
  llvm.func @gpu_shuffles(%val: i32, %id: i32) attributes {intel_reqd_sub_group_size = 32 : i32} {
    %width = arith.constant 16 : i32
    // expected-error@below {{failed to legalize operation 'gpu.shuffle' that was explicitly marked illegal}}
    %shuffleResult, %valid = gpu.shuffle idx %val, %id, %width : i32
    llvm.return
  }
}

// -----

// Cannot convert due to variable shuffle width

gpu.module @shuffles_mismatch {
  llvm.func @gpu_shuffles(%val: i32, %id: i32, %width: i32) attributes {intel_reqd_sub_group_size = 32 : i32} {
    // expected-error@below {{failed to legalize operation 'gpu.shuffle' that was explicitly marked illegal}}
    %shuffleResult, %valid = gpu.shuffle idx %val, %id, %width : i32
    llvm.return
  }
}

// -----

// Cannot convert due to value type not being supported by the conversion

gpu.module @not_supported_lowering {
  llvm.func @gpu_shuffles(%val: f128, %id: i32) attributes {intel_reqd_sub_group_size = 32 : i32} {
    %width = arith.constant 32 : i32
    // expected-error@below {{failed to legalize operation 'gpu.shuffle' that was explicitly marked illegal}}
    %shuffleResult, %valid = gpu.shuffle xor %val, %id, %width : f128
    llvm.return
  }
}


// -----

gpu.module @kernels {
  // CHECK:   llvm.func spir_funccc @no_kernel() {
  gpu.func @no_kernel() {
    gpu.return
  }

  // CHECK:   llvm.func spir_kernelcc @kernel_no_arg() attributes {gpu.kernel} {
  gpu.func @kernel_no_arg() kernel {
    gpu.return
  }

  // CHECK:   llvm.func spir_kernelcc @kernel_with_args(%{{.*}}: f32, %{{.*}}: i64) attributes {gpu.kernel} {
  gpu.func @kernel_with_args(%arg0: f32, %arg1: i64) kernel {
    gpu.return
  }

  // CHECK-64:   llvm.func spir_kernelcc @kernel_with_conv_args(%{{.*}}: i64, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: i64) attributes {gpu.kernel} {
  // CHECK-32:   llvm.func spir_kernelcc @kernel_with_conv_args(%{{.*}}: i32, %{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: i32) attributes {gpu.kernel} {
  gpu.func @kernel_with_conv_args(%arg0: index, %arg1: memref<index>) kernel {
    gpu.return
  }

  // CHECK-64:   llvm.func spir_kernelcc @kernel_with_sized_memref(%{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64) attributes {gpu.kernel} {
  // CHECK-32:   llvm.func spir_kernelcc @kernel_with_sized_memref(%{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) attributes {gpu.kernel} {
  gpu.func @kernel_with_sized_memref(%arg0: memref<1xindex>) kernel {
    gpu.return
  }

  // CHECK-64:   llvm.func spir_kernelcc @kernel_with_ND_memref(%{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64, %{{.*}}: i64) attributes {gpu.kernel} {
  // CHECK-32:   llvm.func spir_kernelcc @kernel_with_ND_memref(%{{.*}}: !llvm.ptr<1>, %{{.*}}: !llvm.ptr<1>, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) attributes {gpu.kernel} {
  gpu.func @kernel_with_ND_memref(%arg0: memref<128x128x128xindex>) kernel {
    gpu.return
  }
}

// -----

gpu.module @kernels {
// CHECK-LABEL:   llvm.func spir_kernelcc @kernel_with_private_attributions() attributes {gpu.kernel} {

// Private attribution is converted to an llvm.alloca

// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x f32 : (i64) -> !llvm.ptr

// MemRef descriptor built from allocated pointer

// CHECK-64:        %[[VAL_4:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-32:        %[[VAL_4:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>

// CHECK:           %[[VAL_5:.*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_4]][0]
// CHECK:           llvm.insertvalue %[[VAL_3]], %[[VAL_5]][1]

// Same code as above

// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           %[[VAL_15:.*]] = llvm.alloca %[[VAL_14]] x i16 : (i64) -> !llvm.ptr

// CHECK-64:        %[[VAL_16:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-32:        %[[VAL_16:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<1 x i32>, array<1 x i32>)>

// CHECK:           %[[VAL_17:.*]] = llvm.insertvalue %[[VAL_15]], %[[VAL_16]][0]
// CHECK:           llvm.insertvalue %[[VAL_15]], %[[VAL_17]][1]
  gpu.func @kernel_with_private_attributions()
      private(%arg2: memref<32xf32, #gpu.address_space<private>>, %arg3: memref<16xi16, #gpu.address_space<private>>)
      kernel {
    gpu.return
  }

// Workgroup attributions are converted to an llvm.ptr<3> argument

// CHECK-LABEL:   llvm.func spir_kernelcc @kernel_with_workgoup_attributions(
// CHECK-SAME:                                                               %[[VAL_29:.*]]: !llvm.ptr<3> {llvm.noalias, llvm.workgroup_attribution = #llvm.mlir.workgroup_attribution<32 : i64, f32>},
// CHECK-SAME:                                                               %[[VAL_30:.*]]: !llvm.ptr<3> {llvm.noalias, llvm.workgroup_attribution = #llvm.mlir.workgroup_attribution<16 : i64, i16>}) attributes {gpu.kernel} {

// MemRef descriptor built from new argument

// CHECK-64:        %[[VAL_31:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-32:        %[[VAL_31:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>

// CHECK:           %[[VAL_32:.*]] = llvm.insertvalue %[[VAL_29]], %[[VAL_31]][0]
// CHECK:           llvm.insertvalue %[[VAL_29]], %[[VAL_32]][1]

// Same as above

// CHECK-64:        %[[VAL_41:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-32:        %[[VAL_41:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<3>, ptr<3>, i32, array<1 x i32>, array<1 x i32>)>

// CHECK:           %[[VAL_42:.*]] = llvm.insertvalue %[[VAL_30]], %[[VAL_41]][0]
// CHECK:           llvm.insertvalue %[[VAL_30]], %[[VAL_42]][1]
  gpu.func @kernel_with_workgoup_attributions()
      workgroup(%arg2: memref<32xf32, #gpu.address_space<workgroup>>, %arg3: memref<16xi16, #gpu.address_space<workgroup>>)
      kernel {
    gpu.return
  }

// Check with both private and workgroup attributions. Simply check additional
// arguments and a llvm.alloca are present.

// CHECK-LABEL:   llvm.func spir_kernelcc @kernel_with_both_attributions(
// CHECK-SAME:                                                           %{{.*}}: !llvm.ptr<3> {llvm.noalias, llvm.workgroup_attribution = #llvm.mlir.workgroup_attribution<8 : i64, f32>},
// CHECK-64-SAME:                                                        %{{.*}}: !llvm.ptr<3> {llvm.noalias, llvm.workgroup_attribution = #llvm.mlir.workgroup_attribution<16 : i64, i64>}) attributes {gpu.kernel} {
// CHECK-32-SAME:                                                        %{{.*}}: !llvm.ptr<3> {llvm.noalias, llvm.workgroup_attribution = #llvm.mlir.workgroup_attribution<16 : i64, i32>}) attributes {gpu.kernel} {

// CHECK:           %[[VAL_79:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:           %[[VAL_80:.*]] = llvm.alloca %[[VAL_79]] x i32 : (i64) -> !llvm.ptr

// CHECK:           %[[VAL_91:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK-64:        %[[VAL_92:.*]] = llvm.alloca %[[VAL_91]] x i64 : (i64) -> !llvm.ptr
// CHECK-32:        %[[VAL_92:.*]] = llvm.alloca %[[VAL_91]] x i32 : (i64) -> !llvm.ptr
  gpu.func @kernel_with_both_attributions()
      workgroup(%arg4: memref<8xf32, #gpu.address_space<workgroup>>, %arg5: memref<16xindex, #gpu.address_space<workgroup>>)
      private(%arg6: memref<32xi32, #gpu.address_space<private>>, %arg7: memref<32xindex, #gpu.address_space<private>>)
      kernel {
    gpu.return
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @kernel_known_block_size
// CHECK-SAME:                                                     reqd_work_group_size = array<i32: 128, 128, 256>
  gpu.func @kernel_known_block_size() kernel attributes {known_block_size = array<i32: 128, 128, 256>} {
    gpu.return
  }
}

// -----

gpu.module @kernels {
// CHECK-LABEL:   llvm.func spir_funccc @address_spaces(
// CHECK-SAME:                                          {{.*}}: !llvm.ptr<1>
// CHECK-SAME:                                          {{.*}}: !llvm.ptr<3>
// CHECK-SAME:                                          {{.*}}: !llvm.ptr
  gpu.func @address_spaces(%arg0: memref<f32, #gpu.address_space<global>>, %arg1: memref<f32, #gpu.address_space<workgroup>>, %arg2: memref<f32, #gpu.address_space<private>>) {
    gpu.return
  }
}

// -----

gpu.module @kernels {
// CHECK:         llvm.func spir_funccc @_Z12get_group_idj(i32)
// CHECK-LABEL:   llvm.func spir_funccc @no_address_spaces(
// CHECK-SAME:                                             %{{[a-zA-Z_][a-zA-Z0-9_]*}}: !llvm.ptr<1>
// CHECK-SAME:                                             %{{[a-zA-Z_][a-zA-Z0-9_]*}}: !llvm.ptr<1>
// CHECK-SAME:                                             %{{[a-zA-Z_][a-zA-Z0-9_]*}}: !llvm.ptr<1>
  gpu.func @no_address_spaces(%arg0: memref<f32>, %arg1: memref<f32, #gpu.address_space<global>>, %arg2: memref<f32>) {
    gpu.return
  }

// CHECK-LABEL:   llvm.func spir_kernelcc @no_address_spaces_complex(
// CHECK-SAME:                                             %{{[a-zA-Z_][a-zA-Z0-9_]*}}: !llvm.ptr<1>
// CHECK-SAME:                                             %{{[a-zA-Z_][a-zA-Z0-9_]*}}: !llvm.ptr<1>
// CHECK:         func.call @no_address_spaces_callee(%{{[0-9]+}}, %{{[0-9]+}})
// CHECK-SAME:                                        : (memref<2x2xf32, 1>, memref<4xf32, 1>)
  gpu.func @no_address_spaces_complex(%arg0: memref<2x2xf32>, %arg1: memref<4xf32>) kernel {
    func.call @no_address_spaces_callee(%arg0, %arg1) : (memref<2x2xf32>, memref<4xf32>) -> ()
    gpu.return
  }
// CHECK-LABEL:   func.func @no_address_spaces_callee(
// CHECK-SAME:                                             [[ARG0:%.*]]: memref<2x2xf32, 1> 
// CHECK-SAME:                                             [[ARG1:%.*]]: memref<4xf32, 1>
// CHECK:         [[C0:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         [[I0:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[C0]]) {
// CHECK-32:         [[I1:%.*]] = builtin.unrealized_conversion_cast [[I0]] : i32 to index
// CHECK-64:         [[I1:%.*]] = builtin.unrealized_conversion_cast [[I0]] : i64 to index
// CHECK:         [[LD:%.*]] = memref.load [[ARG0]]{{\[}}[[I1]], [[I1]]{{\]}} : memref<2x2xf32, 1>
// CHECK:         memref.store [[LD]], [[ARG1]]{{\[}}[[I1]]{{\]}} : memref<4xf32, 1>
  func.func @no_address_spaces_callee(%arg0: memref<2x2xf32>, %arg1: memref<4xf32>) {
    %block_id = gpu.block_id x
    %0 = memref.load %arg0[%block_id, %block_id] : memref<2x2xf32>
    memref.store %0, %arg1[%block_id] : memref<4xf32>
    func.return
  }
}

// -----

// Lowering of subgroup query operations

// CHECK-DAG: llvm.func spir_funccc @_Z18get_sub_group_size() -> i32 attributes {no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z18get_num_sub_groups() -> i32 attributes {no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z22get_sub_group_local_id() -> i32 attributes {no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z16get_sub_group_id() -> i32 attributes {no_unwind, will_return}


gpu.module @subgroup_operations {
// CHECK-LABEL: @gpu_subgroup
  func.func @gpu_subgroup() {
    // CHECK:       %[[SG_ID:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {no_unwind, will_return} : () -> i32
    // CHECK-32-NOT:                llvm.zext
    // CHECK-64           %{{.*}} = llvm.zext %[[SG_ID]] : i32 to i64
    %0 = gpu.subgroup_id : index
    // CHECK: %[[SG_LOCAL_ID:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {no_unwind, will_return}  : () -> i32
    // CHECK-32-NOT:                llvm.zext
    // CHECK-64:          %{{.*}} = llvm.zext %[[SG_LOCAL_ID]] : i32 to i64
    %1 = gpu.lane_id
    // CHECK:     %[[NUM_SGS:.*]] = llvm.call spir_funccc @_Z18get_num_sub_groups() {no_unwind, will_return} : () -> i32
    // CHECK-32-NOT:                llvm.zext
    // CHECK-64:          %{{.*}} = llvm.zext %[[NUM_SGS]] : i32 to i64
    %2 = gpu.num_subgroups : index
    // CHECK:     %[[SG_SIZE:.*]] = llvm.call spir_funccc @_Z18get_sub_group_size() {no_unwind, will_return} : () -> i32
    // CHECK-32-NOT:                llvm.zext
    // CHECK-64:          %{{.*}} = llvm.zext %[[SG_SIZE]] : i32 to i64
    %3 = gpu.subgroup_size : index
    return
  }
}
