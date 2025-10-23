// RUN: mlir-opt -convert-xegpu-to-xevm --split-input-file %s | FileCheck %s

// This file contains tests for materalization patterns added to handle custom type conversions
// added on top of LLVM type converter.

gpu.module @materializecast {
  // CHECK-LABEL: gpu.func @materialize_memref
  // CHECK-SAME: %[[ARG0:.*]]: memref<128xf32>
  gpu.func @materialize_memref(%src: memref<128xf32>) kernel {
    // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<128xf32> -> index
    // CHECK: %[[CASTED:.*]] = arith.index_castui %[[INTPTR]] : index to i64
    %offset = arith.constant dense<0> : vector<1xindex>
    %src_tdesc = xegpu.create_tdesc %src, %offset : memref<128xf32>, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
}

// -----
gpu.module @materializecast {
  // CHECK-LABEL: gpu.func @materialize_ui64
  // CHECK-SAME: %[[ARG0:.*]]: ui64
  gpu.func @materialize_ui64(%src: ui64) kernel {
    // CHECK: %[[VAR0:.*]] = index.castu %[[ARG0]] : ui64 to index
    // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i64
    %offset = arith.constant dense<0> : vector<1xindex>
    %src_tdesc = xegpu.create_tdesc %src, %offset : ui64, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
}

// -----
gpu.module @materializecast {
  // CHECK-LABEL: gpu.func @materialize_ui32
  // CHECK-SAME: %[[ARG0:.*]]: ui32
  gpu.func @materialize_ui32(%src: ui32) kernel {
    // CHECK: %[[VAR0:.*]] = index.castu %[[ARG0]] : ui32 to index
    // CHECK: %[[VAR1:.*]] = arith.index_castui %[[VAR0]] : index to i32
    %offset = arith.constant dense<0> : vector<1xindex>
    %src_tdesc = xegpu.create_tdesc %src, %offset : ui32, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
}

// -----
gpu.module @materializecast {
  // CHECK-LABEL: gpu.func @materialize_single_index_vector
  // CHECK-SAME: %[[ARG0:.*]]: memref<128xf32>
  gpu.func @materialize_single_index_vector(%src: memref<128xf32>) kernel {
    // CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
    // CHECK: %[[VAR1:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
    // CHECK: %[[VAR2:.*]] = arith.index_castui %[[VAR1]] : index to i64
    %offset = arith.constant dense<0> : vector<1xindex>
    %src_tdesc = xegpu.create_tdesc %src, %offset : memref<128xf32>, vector<1xindex>
        -> !xegpu.tensor_desc<1xf32, #xegpu.scatter_tdesc_attr<>>
    gpu.return
  }
}

// -----
gpu.module @materializecast {
  // CHECK-LABEL: gpu.func @materialize_single_elem_vector
  // CHECK-SAME: %[[ARG0:.*]]: memref<128xf32>
  gpu.func @materialize_single_elem_vector(%src: memref<128xf32>) kernel {
    // CHECK: %[[CST:.*]] = arith.constant dense<true> : vector<1xi1>
    // CHECK: %[[VAR1:.*]] = vector.extract %[[CST]][0] : i1 from vector<1xi1>
    %mask = arith.constant dense<1>: vector<1xi1>
    %offset = arith.constant dense<0> : vector<1xindex>
    %0 = xegpu.load %src[%offset], %mask <{chunk_size=8, l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<uncached>}>
      : memref<128xf32>, vector<1xindex>, vector<1xi1> -> vector<8xf32>
    gpu.return
  }
}
