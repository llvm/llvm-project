// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @prefetch_check {
    // CHECK-LABEL: gpu.func @prefetch_matrix_a
    gpu.func @prefetch_matrix_a(%src: memref<16x128xi4, 1>) kernel {
        // CHECK: %[[C64_I32:.*]] = arith.constant 64 : i32
        // CHECK: %[[C8_I32:.*]] = arith.constant 8 : i32
        // CHECK: %[[C16_I32:.*]] = arith.constant 16 : i32
        %srcce = memref.memory_space_cast %src : memref<16x128xi4, 1> to memref<16x128xi4>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<16x128xi4> -> !xegpu.tensor_desc<8x64xi4>

        // CHECK: xevm.blockprefetch2d %{{.*}}, %[[C64_I32]], %[[C16_I32]], %[[C64_I32]], %[[C16_I32]], %[[C8_I32]]
        // CHECK-SAME:  <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>, elem_size_in_bits = 16 : i32,
        // CHECK-SAME:    tile_height = 8 : i32, tile_width = 16 : i32, v_blocks = 1 : i32}> : (!llvm.ptr<1>
        xegpu.prefetch_nd %src_tdesc[8, 64] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x64xi4>

        gpu.return
    }
}
