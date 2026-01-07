// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @load_store_check {
    // CHECK-LABEL: gpu.func @load_store_matrix_a
    // CHECK-SAME: %[[ARG0:.*]]: memref<16x128xi4, 1>, %[[ARG1:.*]]: memref<16x128xi4, 1>
    gpu.func @load_store_matrix_a(%src: memref<16x128xi4, 1>, %dst: memref<16x128xi4, 1>) kernel {
        // CHECK: %[[C64_I32:.*]] = arith.constant 64 : i32
        // CHECK: %[[C8_I32:.*]] = arith.constant 8 : i32
        // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<4xi64>
        // CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
        // CHECK: %[[C128_I32:.*]] = arith.constant 128 : i32
        // CHECK: %[[SRCCE:.*]] = memref.memory_space_cast %[[ARG0]]
        // CHECK: %[[SRCINDEX:.*]] = memref.extract_aligned_pointer_as_index %[[SRCCE]]
        // CHECK: %[[SRCPTR64:.*]] = arith.index_castui %[[SRCINDEX]] : index to i64
        %srcce = memref.memory_space_cast %src : memref<16x128xi4, 1> to memref<16x128xi4>
        // CHECK: %[[DSTTE:.*]] = memref.memory_space_cast %[[ARG1]]
        // CHECK: %[[DSTINDEX:.*]] = memref.extract_aligned_pointer_as_index %[[DSTTE]]
        // CHECK: %[[DSTPTR64:.*]] = arith.index_castui %[[DSTINDEX]] : index to i64
        %dstte = memref.memory_space_cast %dst : memref<16x128xi4, 1> to memref<16x128xi4>

        // CHECK: %[[PAYLOAD_SRC:.*]] = vector.insert %[[SRCPTR64]], %[[CST]] [0] : i64 into vector<4xi64>
        // CHECK: %[[BITCAST1_SRC:.*]] = vector.bitcast %[[PAYLOAD_SRC]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[PAYLOAD1_SRC:.*]] = vector.insert %[[C128_I32]], %[[BITCAST1_SRC]] [2] : i32 into vector<8xi32>
        // CHECK: %[[PAYLOAD2_SRC:.*]] = vector.insert %[[C16_I32]], %[[PAYLOAD1_SRC]] [3] : i32 into vector<8xi32>
        // CHECK: %[[PAYLOAD3_SRC:.*]] = vector.insert %[[C128_I32]], %[[PAYLOAD2_SRC]] [4] : i32 into vector<8xi32>
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<16x128xi4> -> !xegpu.tensor_desc<8x64xi4>

        // CHECK: %[[BITCAST2:.*]] = vector.bitcast %[[PAYLOAD3_SRC]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[SRCPTR64:.*]] = vector.extract %[[BITCAST2]][0] : i64 from vector<4xi64>
        // CHECK: %[[SRCLLVMPTR:.*]] = llvm.inttoptr %[[SRCPTR64]] : i64 to !llvm.ptr<1>
        // CHECK: %[[LOADED:.*]] = xevm.blockload2d %[[SRCLLVMPTR]], %[[C64_I32]],
        // CHECK-SAME: %[[C16_I32]], %[[C64_I32]], %[[C16_I32]], %[[C8_I32]] <{
        // CHECK-SAME:   cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>, elem_size_in_bits = 16 : i32,
        // CHECK-SAME:   pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false,
        // CHECK-SAME:   v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
        %loaded = xegpu.load_nd %src_tdesc[8, 64] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x64xi4> -> vector<32xi4>

        // CHECK: %[[PAYLOAD_DST:.*]] = vector.insert %[[DSTPTR64]], %[[CST]] [0] : i64 into vector<4xi64>
        // CHECK: %[[BITCAST1_DST:.*]] = vector.bitcast %[[PAYLOAD_DST]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[PAYLOAD1_DST:.*]] = vector.insert %[[C128_I32]], %[[BITCAST1_DST]] [2] : i32 into vector<8xi32>
        // CHECK: %[[PAYLOAD2_DST:.*]] = vector.insert %[[C16_I32]], %[[PAYLOAD1_DST]] [3] : i32 into vector<8xi32>
        // CHECK: %[[PAYLOAD3_DST:.*]] = vector.insert %[[C128_I32]], %[[PAYLOAD2_DST]] [4] : i32 into vector<8xi32>
        %dst_tdesc = xegpu.create_nd_tdesc %dstte : memref<16x128xi4> -> !xegpu.tensor_desc<8x64xi4, #xegpu.block_tdesc_attr<memory_space = global>>

        // CHECK: %[[BITCAST2_DST:.*]] = vector.bitcast %[[PAYLOAD3_DST]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[DSTPTR64:.*]] = vector.extract %[[BITCAST2_DST]][0] : i64 from vector<4xi64>
        // CHECK: %[[DSTLLVMPTR:.*]] = llvm.inttoptr %[[DSTPTR64]] : i64 to !llvm.ptr<1>
        // CHECK: xevm.blockstore2d %[[DSTLLVMPTR]], %[[C64_I32]], %[[C16_I32]],
        // CHECK-SAME:  %[[C64_I32]], %[[C16_I32]], %[[C8_I32]], %[[LOADED]] <{
        // CHECK-SAME:    cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>, elem_size_in_bits = 16 : i32,
        // CHECK-SAME:    tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)
        xegpu.store_nd %loaded, %dst_tdesc[8, 64] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<32xi4>, !xegpu.tensor_desc<8x64xi4, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }

    // CHECK-LABEL: gpu.func @load_matrix_b_request_pack
    gpu.func @load_matrix_b_request_pack(%src: memref<64x128xi4, 1>, %dst: memref<64x128xi4, 1>) kernel {
        // CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
        // CHECK: %[[C32_I32:.*]] = arith.constant 32 : i32
        // CHECK: %[[C64_I32:.*]] = arith.constant 64 : i32
        %srcce = memref.memory_space_cast %src : memref<64x128xi4, 1> to memref<64x128xi4>
        %dstte = memref.memory_space_cast %dst : memref<64x128xi4, 1> to memref<64x128xi4>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<64x128xi4> -> !xegpu.tensor_desc<32x32xi4>

        // CHECK: xevm.blockload2d %{{.*}}, %[[C64_I32]], %[[C64_I32]], %[[C64_I32]], %[[C16_I32]], %[[C32_I32]] <{
        // CHECK-SAME:   cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>, elem_size_in_bits = 8 : i32,
        // CHECK-SAME:   pack_register = true, tile_height = 32 : i32, tile_width = 16 : i32, transpose = false,
        // CHECK-SAME:   v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %loaded = xegpu.load_nd %src_tdesc[32, 32] <{packed, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<32x32xi4> -> vector<64xi4>

        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        vector.store %loaded, %dstte[%c32, %c0] : memref<64x128xi4>, vector<64xi4>
        gpu.return
    }
}
