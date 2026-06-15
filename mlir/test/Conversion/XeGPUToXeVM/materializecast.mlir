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

// -----
gpu.module @materializecast [#xevm.target<chip = "pvc">]{
  // CHECK-LABEL: gpu.func @materialize_element_to_2D_single_element_vector
  gpu.func @materialize_element_to_2D_single_element_vector(%dst: memref<128xf32>) kernel {
    %c0 = arith.constant 0 : index
    %alloca_11 = memref.alloca() : memref<512xi8, 3>
    %49 = xegpu.create_mem_desc %alloca_11 : memref<512xi8, 3> -> !xegpu.mem_desc<8x16xf32>
    // CHECK: %[[LOAD:.*]] = llvm.load %{{.+}} : !llvm.ptr<3> -> f32
    // CHECK: %[[BCST:.*]] = vector.broadcast %[[LOAD]] : f32 to vector<1x1xf32>
    %50 = xegpu.load_matrix %49[%c0, %c0] : !xegpu.mem_desc<8x16xf32>, index, index -> vector<1x1xf32>
    %51 = vector.shape_cast %50 : vector<1x1xf32> to vector<1xf32>
    vector.store %51, %dst[%c0] : memref<128xf32>, vector<1xf32>
    gpu.return
  }
}

// -----
gpu.module @materializecast [#xevm.target<chip = "pvc">]{
  // CHECK-LABEL: gpu.func @materialize_2D_single_element_vector_to_element
  gpu.func @materialize_2D_single_element_vector_to_element(%dst: memref<512xi8, 3>) kernel {
    %c0 = arith.constant 0 : index
    %c0_f32 = arith.constant dense<0.0> : vector<1x1xf32>
    %49 = xegpu.create_mem_desc %dst : memref<512xi8, 3> -> !xegpu.mem_desc<8x16xf32>
    // CHECK: %[[EXTR:.*]] = vector.extract %{{.+}}[0, 0] : f32 from vector<1x1xf32>
    // CHECK: llvm.store %[[EXTR]], %{{.+}} : f32, !llvm.ptr<3>
    xegpu.store_matrix %c0_f32, %49[%c0, %c0] : vector<1x1xf32>, !xegpu.mem_desc<8x16xf32>, index, index
    gpu.return
  }
}
