// RUN: mlir-opt --convert-xegpu-to-xevm %s -split-input-file -verify-diagnostics

// Verify that xegpu.dpas with unsupported element types (i16) is rejected
// during XeGPUToXeVM conversion rather than crashing.

gpu.module @test_kernel [#xevm.target<chip = "pvc">] {
  func.func @main() {
    %0 = arith.constant dense<0> : vector<4xi16>
    %1 = arith.constant dense<0> : vector<4xi32>
    // expected-error@+1 {{failed to legalize operation 'xegpu.dpas' that was explicitly marked illegal}}
    %2 = xegpu.dpas %0, %0, %1 : vector<4xi16>, vector<4xi16>, vector<4xi32> -> vector<4xi32>
    return
  }
}

// -----

// Verify that xegpu.store with a memref memory space attribute that has no
// known numeric address space (a SPIR-V storage class, here) is rejected
// during XeGPUToXeVM conversion rather than crashing.

gpu.module @test_kernel {
  gpu.func @store_scatter_unsupported_memspace(%src: memref<1024xf32, #spirv.storage_class<StorageBuffer>>, %offset: vector<1xindex>, %mask: vector<1xi1>) {
    %0 = arith.constant dense<2.9> : vector<1xf32>
    // expected-error@+1 {{failed to legalize operation 'xegpu.store' that was explicitly marked illegal}}
    xegpu.store %0, %src[%offset], %mask <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
        : vector<1xf32>, memref<1024xf32, #spirv.storage_class<StorageBuffer>>, vector<1xindex>, vector<1xi1>
    gpu.return
  }
}

// -----

// Verify that xegpu.lane_shuffle of a sub-byte element type is rejected: the
// shuffle redistributes whole bytes between the lanes, so fp4 fragments cannot
// be shuffled.

gpu.module @test_kernel {
  gpu.func @lane_shuffle_f4(%a: vector<4xf4E2M1FN>) {
    // expected-error@+1 {{failed to legalize operation 'xegpu.lane_shuffle' that was explicitly marked illegal}}
    %0 = xegpu.lane_shuffle %a pack : vector<4xf4E2M1FN>
    gpu.return
  }
}

// -----

// Verify that a xegpu.lane_shuffle fragment wider than 64 bits is rejected.

gpu.module @test_kernel {
  gpu.func @lane_shuffle_too_wide(%a: vector<4xi32>) {
    // expected-error@+1 {{failed to legalize operation 'xegpu.lane_shuffle' that was explicitly marked illegal}}
    %0 = xegpu.lane_shuffle %a pack : vector<4xi32>
    gpu.return
  }
}

// -----

// A non-unit leading dim cannot be lowered to a 2D-block op.

gpu.module @test_kernel {
  gpu.func @load_nd_non_unit_batch(%src: memref<4x8x16xf32>, %z: index) kernel {
    %c0 = arith.constant 0 : index
    %t = xegpu.create_nd_tdesc %src : memref<4x8x16xf32> -> !xegpu.tensor_desc<2x8x16xf32>
    // expected-error@+1 {{failed to legalize operation 'xegpu.load_nd' that was explicitly marked illegal}}
    %v = xegpu.load_nd %t[%z, %c0, %c0] : !xegpu.tensor_desc<2x8x16xf32> -> vector<16xf32>
    gpu.return
  }
}

// -----

// The payload has room for only 3 leading strides, so rank > 5 is rejected.

gpu.module @test_kernel {
  gpu.func @create_nd_tdesc_rank_too_large(%src: memref<2x2x2x2x8x16xf32>) kernel {
    // expected-error@+1 {{failed to legalize operation 'xegpu.create_nd_tdesc' that was explicitly marked illegal}}
    %t = xegpu.create_nd_tdesc %src : memref<2x2x2x2x8x16xf32> -> !xegpu.tensor_desc<1x1x1x1x8x16xf32>
    gpu.return
  }
}

// -----

// A memref with gaps between planes has no exact flattened-plane view.

gpu.module @test_kernel {
  gpu.func @create_nd_tdesc_plane_gap(%src: memref<2x8x16xf32, strided<[200, 16, 1]>>) kernel {
    // expected-error@+1 {{failed to legalize operation 'xegpu.create_nd_tdesc' that was explicitly marked illegal}}
    %t = xegpu.create_nd_tdesc %src : memref<2x8x16xf32, strided<[200, 16, 1]>> -> !xegpu.tensor_desc<1x8x16xf32>
    gpu.return
  }
}

// -----

// Same check applies to an integer source, whose strides are explicit.

gpu.module @test_kernel {
  gpu.func @create_nd_tdesc_plane_gap_ptr(%ptr: i64) kernel {
    // expected-error@+1 {{failed to legalize operation 'xegpu.create_nd_tdesc' that was explicitly marked illegal}}
    %t = xegpu.create_nd_tdesc %ptr, shape: [2, 8, 16], strides: [200, 16, 1] : i64 -> !xegpu.tensor_desc<1x8x16xf32>
    gpu.return
  }
}
