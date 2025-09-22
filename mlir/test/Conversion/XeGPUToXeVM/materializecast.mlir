// RUN: mlir-opt -convert-xegpu-to-xevm --split-input-file %s | FileCheck %s

// This file contains tests for materalization patterns added to handle custom type conversions
// added on top of LLVM type converter.

gpu.module @materializecast {
  // CHECK-LABEL: gpu.func @materialize_memref
  // CHECK-SAME: %[[ARG0:.*]]: memref<128xf32>
  gpu.func @materialize_memref(%src: memref<128xf32>) kernel {
    // CHECK: %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<128xf32> -> index
    // CHECK: %[[CASTED:.*]] = arith.index_castui %[[INTPTR]] : index to i64
    %offset = arith.constant 0 : index
    %mask = arith.constant 1 : i1
    %val = xegpu.load %src[%offset], %mask : memref<128xf32>, index, i1 -> f32
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
    %offset = arith.constant 0 : index
    %mask = arith.constant 1 : i1
    %val = xegpu.load %src[%offset], %mask : ui64, index, i1 -> vector<1xf32>
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
    %offset = arith.constant 0 : index
    %mask = arith.constant 1 : i1
    %val = xegpu.load %src[%offset], %mask : ui32, index, i1 -> vector<1xf32>
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
    // CHECK: %[[CST1:.*]] = arith.constant dense<true> : vector<1xi1>
    // CHECK: %[[VAR3:.*]] = vector.extract %[[CST1]][0] : i1 from vector<1xi1>
    %offset = arith.constant dense<0> : vector<1xindex>
    %mask = arith.constant dense<1> : vector<1xi1>
    %val = xegpu.load %src[%offset], %mask : memref<128xf32>, vector<1xindex>, vector<1xi1> -> vector<1xf32>
    gpu.return
  }
}

