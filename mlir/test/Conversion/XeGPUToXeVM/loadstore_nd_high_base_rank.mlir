// RUN: mlir-opt -convert-xegpu-to-xevm -canonicalize %s | FileCheck %s

gpu.module @load_store_check {
  // CHECK-LABEL: gpu.func @load_store
  // CHECK-SAME: %[[ARG0:.*]]: memref<3x3x8x16xf32, 1>, %[[ARG1:.*]]: memref<3x3x8x16xf32, 1>) kernel {
    gpu.func @load_store(%src: memref<3x3x8x16xf32, 1>, %dst: memref<3x3x8x16xf32, 1>) kernel {
      // CHECK: %[[C32_I32:.*]] = arith.constant 32 : i32
      // CHECK: %[[C64_I32:.*]] = arith.constant 64 : i32
      // CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
      // CHECK: %[[C72_I32:.*]] = arith.constant 72 : i32
      // CHECK: %[[MEMSPACECAST:.*]] = memref.memory_space_cast %[[ARG0]] : memref<3x3x8x16xf32, 1> to memref<3x3x8x16xf32>
      // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[MEMSPACECAST]] : memref<3x3x8x16xf32> -> index
      // CHECK: %[[VAR0:.*]] = arith.index_castui %[[INTPTR]] : index to i64
        %srcce = memref.memory_space_cast %src : memref<3x3x8x16xf32, 1> to memref<3x3x8x16xf32>
      // CHECK: %[[MEMSPACECAST_0:.*]] = memref.memory_space_cast %[[ARG1]] : memref<3x3x8x16xf32, 1> to memref<3x3x8x16xf32>
      // CHECK: %[[INTPTR_1:.*]] = memref.extract_aligned_pointer_as_index %[[MEMSPACECAST_0]] : memref<3x3x8x16xf32> -> index
      // CHECK: %[[VAR1:.*]] = arith.index_castui %[[INTPTR_1]] : index to i64
        %dstte = memref.memory_space_cast %dst : memref<3x3x8x16xf32, 1> to memref<3x3x8x16xf32>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<3x3x8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

      // CHECK: %[[VAR2:.*]] = llvm.inttoptr %[[VAR0]] : i64 to !llvm.ptr<1>
      // CHECK: %[[LOADED:.*]] = xevm.blockload2d %[[VAR2]], %[[C64_I32]], %[[C72_I32]], %[[C64_I32]],
      // CHECK-SAME: %[[C0_I32]], %[[C64_I32]] <{cache_control = #xevm.load_cache_control<L1c_L2uc_L3uc>,
      // CHECK-SAME:  elem_size_in_bits = 32 : i32, pack_register = false, tile_height = 8 : i32,
      // CHECK-SAME:  tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32}>
        %loaded = xegpu.load_nd %src_tdesc[2, 2, 0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
      // CHECK: %[[VAR7:.*]] = vector.insert
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

        %dst_tdesc = xegpu.create_nd_tdesc %dstte : memref<3x3x8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

      // CHECK: %[[VAR8:.*]] = llvm.inttoptr %[[VAR1]] : i64 to !llvm.ptr<1>
      // CHECK: %[[VAR9:.*]] = vector.bitcast %[[VAR7]] : vector<8xf32> to vector<8xi32>
      // CHECK: xevm.blockstore2d %[[VAR8]], %[[C64_I32]], %[[C72_I32]], %[[C64_I32]], %[[C0_I32]], %[[C32_I32]], %[[VAR9]]
      // CHECK-SAME: <{cache_control = #xevm.store_cache_control<L1wb_L2uc_L3uc>,
      // CHECK-SAME:  elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 16 : i32}>
        xegpu.store_nd %loaded_modified, %dst_tdesc[1, 1, 0, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }
}
