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
