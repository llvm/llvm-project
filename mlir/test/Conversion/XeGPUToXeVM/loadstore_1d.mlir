// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @load_store_check {
    // CHECK-LABEL: @load_store(
    // CHECK-SAME: %[[SRC:.*]]: memref<8x64xf32, 1>, %[[DST:.*]]: memref<8x32xf32, 1>
    gpu.func @load_store(%src: memref<8x64xf32, 1>, %dst: memref<8x32xf32, 1>) kernel {
        // CHECK: %[[C512:.*]] = arith.constant 512 : i64
        // CHECK: %[[C32:.*]] = arith.constant 32 : i32
        // CHECK: %[[C384:.*]] = arith.constant 384 : i64
        // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<4xi64>
        // CHECK: %[[C8:.*]] = arith.constant 8 : i32
        // CHECK: %[[C64:.*]] = arith.constant 64 : i32
        // CHECK: %[[C0:.*]] = arith.constant 0 : i32

        // CHECK: %[[SRCCE:.*]] = memref.memory_space_cast %[[SRC]] : memref<8x64xf32, 1> to memref<8x64xf32>
        %srcce = memref.memory_space_cast %src : memref<8x64xf32, 1> to memref<8x64xf32>
        // CHECK: %[[DSTTE:.*]] = memref.memory_space_cast %[[DST]] : memref<8x32xf32, 1> to memref<8x32xf32>
        %dstte = memref.memory_space_cast %dst : memref<8x32xf32, 1> to memref<8x32xf32>

        // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[SRCCE]] : memref<8x64xf32> -> index
        // CHECK: %[[INTPTR_I64:.*]] = arith.index_castui %[[INTPTR]] : index to i64
        // CHECK: %[[VEC1:.*]] = vector.insert %[[INTPTR_I64]], %[[CST]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VEC2:.*]] = vector.bitcast %[[VEC1]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VEC3:.*]] = vector.insert %[[C64]], %[[VEC2]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VEC4:.*]] = vector.insert %[[C8]], %[[VEC3]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VEC5:.*]] = vector.insert %[[C0]], %[[VEC4]] [4] : i32 into vector<8xi32>
        // CHECK: %[[VEC6:.*]] = vector.insert %[[C0]], %[[VEC5]] [5] : i32 into vector<8xi32>
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<8x64xf32> -> !xegpu.tensor_desc<32xf32>
        // CHECK: %[[VEC7:.*]] = vector.bitcast %[[VEC6]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[EXTR:.*]] = vector.extract %[[VEC7]][0] : i64 from vector<4xi64>
        // CHECK: %[[ADDR:.*]] = arith.addi %[[EXTR]], %[[C384]] : i64
        // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ADDR]] : i64 to !llvm.ptr<1>
        // CHECK: %[[LOAD:.*]] = xevm.blockload %[[PTR]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>}>
        // CHECK-SAME: : (!llvm.ptr<1>) -> vector<2xi32>
        %loaded = xegpu.load_nd %src_tdesc[1, 32] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<32xf32> -> vector<2xf32>

        // CHECK: %[[INTPTR1:.*]] = memref.extract_aligned_pointer_as_index %[[DSTTE]] : memref<8x32xf32> -> index
        // CHECK: %[[INTPTR1_I64:.*]] = arith.index_castui %[[INTPTR1]] : index to i64
        // CHECK: %[[VEC1_1:.*]] = vector.insert %[[INTPTR1_I64]], %[[CST]] [0] : i64 into vector<4xi64>
        // CHECK: %[[VEC2_1:.*]] = vector.bitcast %[[VEC1_1]] : vector<4xi64> to vector<8xi32>
        // CHECK: %[[VEC3_1:.*]] = vector.insert %[[C32]], %[[VEC2_1]] [2] : i32 into vector<8xi32>
        // CHECK: %[[VEC4_1:.*]] = vector.insert %[[C8]], %[[VEC3_1]] [3] : i32 into vector<8xi32>
        // CHECK: %[[VEC5_1:.*]] = vector.insert %[[C0]], %[[VEC4_1]] [4] : i32 into vector<8xi32>
        // CHECK: %[[VEC6_1:.*]] = vector.insert %[[C0]], %[[VEC5_1]] [5] : i32 into vector<8xi32>
        %dst_tdesc = xegpu.create_nd_tdesc %dstte : memref<8x32xf32> -> !xegpu.tensor_desc<32xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        // CHECK: %[[VEC7_1:.*]] = vector.bitcast %[[VEC6_1]] : vector<8xi32> to vector<4xi64>
        // CHECK: %[[EXTR1:.*]] = vector.extract %[[VEC7_1]][0] : i64 from vector<4xi64>
        // CHECK: %[[ADDR1:.*]] = arith.addi %[[EXTR1]], %[[C512]] : i64
        // CHECK: %[[PTR1:.*]] = llvm.inttoptr %[[ADDR1]] : i64 to !llvm.ptr<1>
        // CHECK: xevm.blockstore %[[PTR1]], %[[LOAD]] <{cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>}>
        // CHECK-SAME: : (!llvm.ptr<1>, vector<2xi32>)
        xegpu.store_nd %loaded, %dst_tdesc[4, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<2xf32>, !xegpu.tensor_desc<32xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }
}
