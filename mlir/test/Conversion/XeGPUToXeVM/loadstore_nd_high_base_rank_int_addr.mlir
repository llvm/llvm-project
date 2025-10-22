// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize -cse %s | FileCheck %s

gpu.module @load_store_check {
  // CHECK-LABEL: gpu.func @load_store
  // CHECK-SAME:  %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64,
  // CHECK-SAME:  %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index, %[[ARG6:.*]]: index, %[[ARG7:.*]]: index, %[[ARG8:.*]]: index, %[[ARG9:.*]]: index
  gpu.func @load_store(%src: i64, %dst: i64, %dim0: index, %dim1: index, %dim2: index, %dim3: index,
                       %stride0: index, %stride1: index, %stride2: index, %stride3: index) kernel {
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
    // CHECK: %[[C4_I32:.*]] = arith.constant 4 : i32
    // CHECK: %[[VAR0:.*]] = arith.index_cast %[[ARG5]] : index to i32
    // CHECK: %[[VAR1:.*]] = arith.index_cast %[[ARG2]] : index to i64
    // CHECK: %[[VAR2:.*]] = arith.index_cast %[[ARG3]] : index to i64
    // CHECK: %[[VAR3:.*]] = arith.muli %[[VAR1]], %[[VAR2]] : i64
    // CHECK: %[[VAR4:.*]] = arith.index_cast %[[ARG4]] : index to i64
    // CHECK: %[[VAR5:.*]] = arith.muli %[[VAR3]], %[[VAR4]] : i64
    // CHECK: %[[VAR6:.*]] = arith.trunci %[[VAR5]] : i64 to i32
    // CHECK: %[[VAR7:.*]] = arith.muli %[[ARG4]], %[[C2]] : index
    // CHECK: %[[VAR8:.*]] = arith.muli %[[ARG4]], %[[ARG3]] : index
    // CHECK: %[[VAR9:.*]] = arith.muli %[[VAR8]], %[[C2]] : index
    // CHECK: %[[VAR10:.*]] = arith.addi %[[VAR7]], %[[VAR9]] : index
    // CHECK: %[[VAR11:.*]] = arith.index_cast %[[VAR10]] : index to i32
    %src_tdesc = xegpu.create_nd_tdesc %src, shape:[%dim0, %dim1, %dim2, %dim3],
                   strides:[%stride0, %stride1, %stride2, %stride3] : i64 -> !xegpu.tensor_desc<8x16xf32>

    // CHECK: %[[SRC_PTR:.*]] = llvm.inttoptr %[[ARG0]] : i64 to !llvm.ptr<1>
    // CHECK: %[[VAR13:.*]] = arith.muli %[[VAR0]], %[[C4_I32]] : i32
    // CHECK: %[[LOADED:.*]] = xevm.blockload2d %[[SRC_PTR]], %[[VAR13]], %[[VAR6]], %[[VAR13]], %[[C0_I32]], %[[VAR11]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>, elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}>
    // CHECK: %[[VAR15:.*]] = vector.bitcast %[[LOADED]] : vector<8xi32> to vector<8xf32>
    %loaded = xegpu.load_nd %src_tdesc[2, 2, 0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

    %tid_x = gpu.thread_id x
    %tid_x_i32 = arith.index_cast %tid_x : index to i32
    %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
    // CHECK: %[[LOADED_MODIFIED:.*]] = vector.insert
    %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

    // CHECK: %[[VAR19:.*]] = arith.addi %[[ARG4]], %[[VAR8]] : index
    // CHECK: %[[VAR20:.*]] = arith.index_cast %[[VAR19]] : index to i32
    %dst_tdesc = xegpu.create_nd_tdesc %dst, shape:[%dim0, %dim1, %dim2, %dim3],
                   strides:[%stride0, %stride1, %stride2, %stride3] : i64 -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

    // CHECK: %[[DST_PTR:.*]] = llvm.inttoptr %[[ARG1]] : i64 to !llvm.ptr<1>
    // CHECK: %[[LOADED_MODIFIED_BITCAST:.*]] = vector.bitcast %[[LOADED_MODIFIED]] : vector<8xf32> to vector<8xi32>
    // CHECK: xevm.blockstore2d %[[DST_PTR]], %[[VAR13]], %[[VAR6]], %[[VAR13]], %[[C0_I32]], %[[VAR20]], %[[LOADED_MODIFIED_BITCAST]] <{cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>, elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}>
    xegpu.store_nd %loaded_modified, %dst_tdesc[1, 1, 0, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
    gpu.return
  }
}
