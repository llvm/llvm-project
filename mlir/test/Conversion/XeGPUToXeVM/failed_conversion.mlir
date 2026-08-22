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

// A 2D-block op transfers the innermost 2 dims only, so a non-unit leading dim
// must be rejected rather than lowered to a load of the wrong shape.

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
