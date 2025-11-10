// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @load_store_check {
    // CHECK-LABEL: @load_store(
    // CHECK-SAME: %[[SRC:.*]]: memref<512xf32, 1>, %[[DST:.*]]: memref<256xf32, 1>
    gpu.func @load_store(%src: memref<512xf32, 1>, %dst: memref<256xf32, 1>) kernel {
        // CHECK: %[[C512:.*]] = arith.constant 512 : i64
        // CHECK: %[[C384:.*]] = arith.constant 384 : i64

        // CHECK: %[[SRCCE:.*]] = memref.memory_space_cast %[[SRC]] : memref<512xf32, 1> to memref<512xf32>
        %srcce = memref.memory_space_cast %src : memref<512xf32, 1> to memref<512xf32>
        // CHECK: %[[DSTTE:.*]] = memref.memory_space_cast %[[DST]] : memref<256xf32, 1> to memref<256xf32>
        %dstte = memref.memory_space_cast %dst : memref<256xf32, 1> to memref<256xf32>

        // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[SRCCE]] : memref<512xf32> -> index
        // CHECK: %[[INTPTR_I64:.*]] = arith.index_castui %[[INTPTR]] : index to i64
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<512xf32> -> !xegpu.tensor_desc<32xf32>
        // CHECK: %[[ADDR:.*]] = arith.addi %[[INTPTR_I64]], %[[C384]] : i64
        // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ADDR]] : i64 to !llvm.ptr<1>
        // CHECK: %[[LOAD:.*]] = xevm.blockload %[[PTR]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}>
        // CHECK-SAME: : (!llvm.ptr<1>) -> vector<2xi32>
        %loaded = xegpu.load_nd %src_tdesc[96] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<32xf32> -> vector<2xf32>

        // CHECK: %[[INTPTR1:.*]] = memref.extract_aligned_pointer_as_index %[[DSTTE]] : memref<256xf32> -> index
        // CHECK: %[[INTPTR1_I64:.*]] = arith.index_castui %[[INTPTR1]] : index to i64
        %dst_tdesc = xegpu.create_nd_tdesc %dstte : memref<256xf32> -> !xegpu.tensor_desc<32xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        // CHECK: %[[ADDR1:.*]] = arith.addi %[[INTPTR1_I64]], %[[C512]] : i64
        // CHECK: %[[PTR1:.*]] = llvm.inttoptr %[[ADDR1]] : i64 to !llvm.ptr<1>
        // CHECK: xevm.blockstore %[[PTR1]], %[[LOAD]] <{cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>}>
        // CHECK-SAME: : (!llvm.ptr<1>, vector<2xi32>)
        xegpu.store_nd %loaded, %dst_tdesc[128] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<2xf32>, !xegpu.tensor_desc<32xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }
}
