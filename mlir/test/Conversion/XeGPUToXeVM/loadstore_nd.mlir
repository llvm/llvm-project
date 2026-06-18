// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @load_store_check {
    // CHECK-LABEL: gpu.func @load_store(
    gpu.func @load_store(%src: memref<8x16xf32, 1>, %dst: memref<8x16xf32, 1>) kernel {
        // CHECK: %[[W_P_BYTES:.*]] = arith.constant 64 : i32
        // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
        // CHECK: %[[H:.*]] = arith.constant 8 : i32
        %srcce = memref.memory_space_cast %src : memref<8x16xf32, 1> to memref<8x16xf32>
        %dstte = memref.memory_space_cast %dst : memref<8x16xf32, 1> to memref<8x16xf32>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

        //CHECK: %[[LD_LOADED_I32:.*]] = xevm.blockload2d %{{.*}}, %[[W_P_BYTES]], %[[H]], %[[W_P_BYTES]], %[[ZERO]], %[[ZERO]]
        //CHECK-SAME: <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3c>, elem_size_in_bits = 32 : i32,
        //CHECK-SAME:   pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false,
        //CHECK-SAME:   v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %loaded = xegpu.load_nd %src_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

        %dst_tdesc = xegpu.create_nd_tdesc %dstte : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

        //CHECK: xevm.blockstore2d %{{.*}}, %[[W_P_BYTES]], %[[H]], %[[W_P_BYTES]], %[[ZERO]], %[[ZERO]], %{{.*}} <{
        //CHECK-SAME:   cache_control = #xevm.store_cache_control<L1wb_L2uc_L3wb>, elem_size_in_bits = 32 : i32,
        //CHECK-SAME:   tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
        xegpu.store_nd %loaded_modified, %dst_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }

    // CHECK-LABEL: gpu.func @load_store_with_partial_cache_hints(
    gpu.func @load_store_with_partial_cache_hints(%src: memref<8x16xf32, 1>, %dst: memref<8x16xf32, 1>) kernel {
        // CHECK: %[[W_P_BYTES:.*]] = arith.constant 64 : i32
        // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
        // CHECK: %[[H:.*]] = arith.constant 8 : i32
        %srcce = memref.memory_space_cast %src : memref<8x16xf32, 1> to memref<8x16xf32>
        %dstte = memref.memory_space_cast %dst : memref<8x16xf32, 1> to memref<8x16xf32>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

        //CHECK: %[[LD_LOADED_I32:.*]] = xevm.blockload2d %{{.*}}, %[[W_P_BYTES]], %[[H]], %[[W_P_BYTES]], %[[ZERO]], %[[ZERO]]
        //CHECK-SAME: <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3c>, elem_size_in_bits = 32 : i32,
        //CHECK-SAME:   pack_register = false, tile_height = 8 : i32, tile_width = 16 : i32, transpose = false,
        //CHECK-SAME:   v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        %loaded = xegpu.load_nd %src_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>}>
            : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

        %dst_tdesc = xegpu.create_nd_tdesc %dstte : memref<8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

        //CHECK: xevm.blockstore2d %{{.*}}, %[[W_P_BYTES]], %[[H]], %[[W_P_BYTES]], %[[ZERO]], %[[ZERO]], %{{.*}} <{
        //CHECK-SAME:   cache_control = #xevm.store_cache_control<Use_Default>, elem_size_in_bits = 32 : i32,
        //CHECK-SAME:   tile_height = 8 : i32, tile_width = 16 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
        xegpu.store_nd %loaded_modified, %dst_tdesc[0, 0] <{}>
            : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }

    // A plain 32x16 8-bit block load with no VNNI/pack and no transpose is not a valid
    // hardware message. But VNNI and non-VNNI loads yield the same data for this specific shape.
    // Conversion forces VNNI (pack_register) is this case:
    // the payload is loaded as i32 and bitcast back to the requested 8-bit type.
    // CHECK-LABEL: gpu.func @load_nd_8bit(
    gpu.func @load_nd_8bit(%src: memref<32x16xi8, 1>, %dst: memref<32x16xi8, 1>) kernel {
        // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
        // CHECK: %[[H:.*]] = arith.constant 32 : i32
        // CHECK: %[[W:.*]] = arith.constant 16 : i32
        %srcce = memref.memory_space_cast %src : memref<32x16xi8, 1> to memref<32x16xi8>
        %dstte = memref.memory_space_cast %dst : memref<32x16xi8, 1> to memref<32x16xi8>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<32x16xi8> -> !xegpu.tensor_desc<32x16xi8>

        // CHECK: %[[LOADED:.*]] = xevm.blockload2d %{{.*}}, %[[W]], %[[H]], %[[W]], %[[ZERO]], %[[ZERO]]
        // CHECK-SAME: <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3c>, elem_size_in_bits = 8 : i32,
        // CHECK-SAME:   pack_register = true, tile_height = 32 : i32, tile_width = 16 : i32, transpose = false,
        // CHECK-SAME:   v_blocks = 1 : i32}> : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi32>
        // CHECK: vector.bitcast %[[LOADED]] : vector<8xi32> to vector<32xi8>
        %loaded = xegpu.load_nd %src_tdesc[0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<32x16xi8> -> vector<32xi8>

        %c0 = arith.constant 0 : index
        vector.store %loaded, %dstte[%c0, %c0] : memref<32x16xi8>, vector<32xi8>
        gpu.return
    }
}
