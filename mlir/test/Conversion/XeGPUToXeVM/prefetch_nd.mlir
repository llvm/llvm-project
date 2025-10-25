// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @prefetch_nd_check {
    // CHECK-LABEL: gpu.func @prefetch_nd(
    // CHECK-SAME: %[[ARG0:.*]]: memref<8x16xf32, 1>, %[[ARG1:.*]]: memref<8x16xf32, 1>) kernel {
    gpu.func @prefetch_nd(%src: memref<8x16xf32, 1>, %dst: memref<8x16xf32, 1>) kernel {
        // CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<8x16xf32, 1> to memref<8x16xf32>
        // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[MEMSPACECAST]] : memref<8x16xf32> -> index
        // CHECK: %[[VAR0:.*]] = arith.index_castui %[[INTPTR]] : index to i64
        %srcce = memref.memory_space_cast %src : memref<8x16xf32, 1> to memref<8x16xf32>
        // CHECK: %[[MEMSPACECAST_0:.*]] = memref.memory_space_cast %[[ARG1]] : memref<8x16xf32, 1> to memref<8x16xf32>
        %dstte = memref.memory_space_cast %dst : memref<8x16xf32, 1> to memref<8x16xf32>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32,
            #xegpu.block_tdesc_attr<memory_space = global>>
        // CHECK: %[[C16_I64:.*]] = arith.constant 16 : i64
        // CHECK: %[[VAR1:.*]] = arith.trunci %[[C16_I64]] : i64 to i32
        // CHECK: %[[C8_I64:.*]] = arith.constant 8 : i64
        // CHECK: %[[VAR2:.*]] = arith.trunci %[[C8_I64]] : i64 to i32
        // CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
        // CHECK: %[[VAR3:.*]] = arith.trunci %[[C0_I64]] : i64 to i32
        // CHECK: %[[C0_I64_1:.*]] = arith.constant 0 : i64
        // CHECK: %[[VAR4:.*]] = arith.trunci %[[C0_I64_1]] : i64 to i32
        // CHECK: %[[VAR5:.*]] = llvm.inttoptr %[[VAR0]] : i64 to !llvm.ptr<1>
        // CHECK: %[[C4_I32:.*]] = arith.constant 4 : i32
        // CHECK: %[[VAR6:.*]] = arith.muli %[[VAR1]], %[[C4_I32]] : i32
        // CHECK: xevm.blockprefetch2d %[[VAR5]], %[[VAR6]], %[[VAR2]], %[[VAR6]], %[[VAR3]], %[[VAR4]]
        // CHECK-SAME: <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>, elem_size_in_bits = 32 : i32,
        // CHECK-SAME:   tile_height = 8 : i32, tile_width = 16 : i32, v_blocks = 1 : i32}>
        xegpu.prefetch_nd %src_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

        gpu.return
    }
}

