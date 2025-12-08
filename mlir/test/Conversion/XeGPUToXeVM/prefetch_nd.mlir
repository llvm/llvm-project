// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @prefetch_nd_check {
    // CHECK-LABEL: gpu.func @prefetch_nd
    gpu.func @prefetch_nd(%src: memref<8x16xf32, 1>, %dst: memref<8x16xf32, 1>) kernel {
        // CHECK: %[[BASE_WIDTH_PITCH_BYTES:.*]] = arith.constant 64 : i32
        // CHECK: %[[OFFSET_ZERO:.*]] = arith.constant 0 : i32
        // CHECK: %[[BASE_H:.*]] = arith.constant 8 : i32
        %srcce = memref.memory_space_cast %src : memref<8x16xf32, 1> to memref<8x16xf32>
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32,
            #xegpu.block_tdesc_attr<memory_space = global>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>

        //CHECK: %[[LLVMPTR:.*]] = llvm.inttoptr %{{.*}} : i64 to !llvm.ptr<1>
        //CHECK: xevm.blockprefetch2d %[[LLVMPTR]], %[[BASE_WIDTH_PITCH_BYTES]], %[[BASE_H]],
        //CHECK-SAME:   %[[BASE_WIDTH_PITCH_BYTES]], %[[OFFSET_ZERO]], %[[OFFSET_ZERO]]
        //CHECK-SAME:   <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>, elem_size_in_bits = 32 : i32,
        //CHECK-SAME:     tile_height = 8 : i32, tile_width = 16 : i32, v_blocks = 1 : i32}>
        //CHECK-SAME:   : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
        xegpu.prefetch_nd %src_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>,
                  #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>

        gpu.return
    }
}

