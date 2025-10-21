// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @load_store_check {
    // CHECK: fail
    gpu.func @load_store(%src: memref<3x3x8x16xf32, 1>, %dst: memref<3x3x8x16xf32, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<3x3x8x16xf32, 1> to memref<3x3x8x16xf32>
        %dstte = memref.memory_space_cast %dst : memref<3x3x8x16xf32, 1> to memref<3x3x8x16xf32>

        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<3x3x8x16xf32> -> !xegpu.tensor_desc<8x16xf32>

        %loaded = xegpu.load_nd %src_tdesc[2, 2, 0, 0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<8x16xf32> -> vector<8xf32>

        %tid_x = gpu.thread_id x
        %tid_x_i32 = arith.index_cast %tid_x : index to i32
        %tid_x_f32 = arith.sitofp %tid_x_i32 : i32 to f32
        %loaded_modified = vector.insert %tid_x_f32, %loaded[0] : f32 into vector<8xf32>

        %dst_tdesc = xegpu.create_nd_tdesc %dstte : memref<3x3x8x16xf32> -> !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>

        xegpu.store_nd %loaded_modified, %dst_tdesc[1, 1, 0, 0] <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
            : vector<8xf32>, !xegpu.tensor_desc<8x16xf32, #xegpu.block_tdesc_attr<memory_space = global>>
        gpu.return
    }
}
