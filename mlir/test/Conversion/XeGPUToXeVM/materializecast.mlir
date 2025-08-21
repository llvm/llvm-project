// RUN: mlir-opt -convert-xegpu-to-xevm %s | FileCheck %s

gpu.module @materializecast {
  // CHECK-LABEL: gpu.func @materialize_memref
  // CHECK-SAME: %[[ARG0:.*]]: memref<128xf32>
  gpu.func @materialize_memref(%src: memref<128xf32>) kernel {
    // CHECK: XXX
    %offset = arith.constant dense<0> : vector<1xindex>
    %src_tdesc = xegpu.create_tdesc %src, %offset : memref<128xf32>, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
  // CHECK-LABEL: gpu.func @materialize_ui64
  // CHECK-SAME: %[[ARG0:.*]]: ui64
  gpu.func @materialize_ui64(%src: ui64) kernel {
    // CHECK: XXX
    %offset = arith.constant dense<0> : vector<1xindex>
    %src_tdesc = xegpu.create_tdesc %src, %offset : ui64, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
  // CHECK-LABEL: gpu.func @materialize_ui32
  // CHECK-SAME: %[[ARG0:.*]]: ui32
  gpu.func @materialize_ui32(%src: ui32) kernel {
    %offset = arith.constant dense<0> : vector<1xindex>
    //%src_tdesc = xegpu.create_tdesc %src, %offset : ui32, vector<1xindex>
    //    -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
  // CHECK-LABEL: gpu.func @materialize_single_index_vector
  // CHECK-SAME: %[[ARG0:.*]]: memref<128xf32>
  gpu.func @materialize_single_index_vector(%src: memref<128xf32>) kernel {
    // CHECK: XXX
    %offset = arith.constant dense<0> : vector<1xindex>
    %src_tdesc = xegpu.create_tdesc %src, %offset : memref<128xf32>, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
  // CHECK-LABEL: gpu.func @materialize_single_elem_vector
  // CHECK-SAME: %[[ARG0:.*]]: vector<1xi1>
  gpu.func @materialize_single_elem_vector(%src: memref<128xf32>) kernel {
    // CHECK: XXX
    %mask = arith.constant dense<1>: vector<1xi1>
    %offset = arith.constant dense<0> : vector<1xindex>
    %0 = xegpu.load %src[%offset], %mask <{chunk_size=8, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : memref<128xf32>, vector<1xindex>, vector<1xi1> -> vector<1x8xf32>
    gpu.return
  }
}
