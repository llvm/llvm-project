// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

// Transpose is a hardware feature of the 2D block load message that is only
// available for 32-bit elements. When a transposed load is requested for a
// sub-32-bit element type, the conversion emulates it by reinterpreting the
// tile as 32-bit elements: the element size is promoted to 32 bits, the tile
// width is scaled down by (32 / elemBitSize), and the column offset (offsetW)
// is right-shifted by log2(32 / elemBitSize) to account for the wider element.

gpu.module @load_check {
    // CHECK-LABEL: gpu.func @load_nd_transpose_f16(
    gpu.func @load_nd_transpose_f16(%src: memref<16x16xf16, 1>, %dst: memref<16x16xf16, 1>) kernel {
        %srcce = memref.memory_space_cast %src : memref<16x16xf16, 1> to memref<16x16xf16>
        %dstte = memref.memory_space_cast %dst : memref<16x16xf16, 1> to memref<16x16xf16>
        %src_tdesc = xegpu.create_nd_tdesc %srcce : memref<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

        // The element byte size used for the surface-width/pitch computation is
        // still the original 16-bit element size (2 bytes).
        // CHECK: %[[ELEM_BYTES:.*]] = arith.constant 2 : i32
        // CHECK: %[[OFFSET_W:.*]] = arith.trunci %{{.*}} : i64 to i32
        // CHECK: %[[OFFSET_H:.*]] = arith.trunci %{{.*}} : i64 to i32

        // 32 / 16 = 2, so offsetW is shifted right by log2(2) = 1.
        // CHECK: %[[SHIFT:.*]] = arith.constant 1 : i32
        // CHECK: %[[OFFSET_W_SCALED:.*]] = arith.shrsi %[[OFFSET_W]], %[[SHIFT]] : i32

        // The block load is issued with 32-bit elements, the tile width scaled
        // from 16 down to 8, the (unchanged) tile height, the transpose request,
        // and the scaled column offset.
        // CHECK: xevm.blockload2d %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[OFFSET_W_SCALED]], %[[OFFSET_H]]
        // CHECK-SAME: elem_size_in_bits = 32 : i32
        // CHECK-SAME: pack_register = false
        // CHECK-SAME: tile_height = 16 : i32
        // CHECK-SAME: tile_width = 8 : i32
        // CHECK-SAME: transpose = true
        // CHECK-SAME: -> vector<8xi32>
        %loaded = xegpu.load_nd %src_tdesc[0, 16] <{transpose = array<i64: 1, 0>, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
            : !xegpu.tensor_desc<16x16xf16> -> vector<16xf16>

        // The loaded i32 payload is bitcast back to the requested f16 type.
        // CHECK: vector.bitcast %{{.*}} : vector<8xi32> to vector<16xf16>
        %c0 = arith.constant 0 : index
        vector.store %loaded, %dstte[%c0, %c0] : memref<16x16xf16>, vector<16xf16>
        gpu.return
    }
}
