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
// known numeric address space (a bare string, here) is rejected during
// XeGPUToXeVM conversion rather than crashing.

gpu.module @test_kernel {
  gpu.func @store_scatter_unsupported_memspace(%src: memref<1024xf32, "foo">, %offset: vector<1xindex>, %mask: vector<1xi1>) {
    %0 = arith.constant dense<2.9> : vector<1xf32>
    // expected-error@+1 {{failed to legalize operation 'xegpu.store' that was explicitly marked illegal}}
    xegpu.store %0, %src[%offset], %mask <{l1_hint = #xegpu.cache_hint<write_back>, l2_hint = #xegpu.cache_hint<uncached>}>
        : vector<1xf32>, memref<1024xf32, "foo">, vector<1xindex>, vector<1xi1>
    gpu.return
  }
}
