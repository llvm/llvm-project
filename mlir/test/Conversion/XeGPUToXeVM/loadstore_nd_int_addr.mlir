// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @load_store_check {
    // CHECK-LABEL: gpu.func @load_store
    // CHECK-SAME: %[[ARG0:.*]]: ui64, %[[ARG1:.*]]: ui32) kernel {
    gpu.func @load_store(%src: ui64, %dst: ui32) kernel {
        // CHECK: %[[C64_I32:.*]] = arith.constant 64 : i32
        // CHECK: %[[C0_I32:.*]] = arith.constant 0
        // CHECK: %[[C8_I32:.*]] = arith.constant 8 : i32
        // CHECK: %[[ARG1_IDX:.*]] = index.castu %[[ARG1]] : ui32 to index
        // CHECK: %[[ARG1_I32:.*]] = arith.index_castui %[[ARG1_IDX]] : index to i32
        // CHECK: %[[ARG0_IDX:.*]] = index.castu %[[ARG0]] : ui64 to index
        // CHECK: %[[ARG0_I64:.*]] = arith.index_castui %[[ARG0_IDX]] : index to i64
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        %src_tdesc = xegpu.create_nd_tdesc %src, shape:[%c8, %c16], strides:[%c16, %c1] : ui64 -> !xegpu.tensor_desc<8x16xf32>


        // CHECK: %[[VAR4:.*]] = llvm.inttoptr %[[ARG0_I64]] : i64 to !llvm.ptr<1>
        // CHECK: %[[LOAD:.*]] = xevm.blockload2d %[[VAR4]], %[[C64_I32]], %[[C8_I32]], %[[C64_I32]],
        // CHECK-SAME:  %[[C0_I32]], %[[C0_I32]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>,
        // CHECK-SAME:  elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32,
        // CHECK-SAME:  tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}>
        // CHECK: %[[VAR6:.*]] = vector.bitcast %[[LOAD]] : vector<8xi32> to vector<8xf32>
        %loaded = xegpu.load_nd %src_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        // CHECK: %[[VAR9:.*]] = vector.insert
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

        // CHECK: %[[VAR10:.*]] = arith.extui %[[ARG1_I32]] : i32 to i64
        %dst_tdesc = xegpu.create_nd_tdesc %dst, shape:[%c8, %c16], strides:[%c16, %c1] : ui32 -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

        // CHECK: %[[VAR11:.*]] = llvm.inttoptr %[[VAR10]] : i64 to !llvm.ptr<1>
        // CHECK: %[[STORE:.*]] = vector.bitcast %[[VAR9]] : vector<8xf32> to vector<8xi32>
        // CHECK: xevm.blockstore2d %[[VAR11]], %[[C64_I32]], %[[C8_I32]], %[[C64_I32]], %[[C0_I32]], %[[C0_I32]], %[[STORE]]
        // CHECK-SAME: <{cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>,
        // CHECK-SAME:  elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}>
        xegpu.store_nd %loaded_modified, %dst_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }
}
