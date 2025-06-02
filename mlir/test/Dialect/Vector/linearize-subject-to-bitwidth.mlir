// RUN: mlir-opt %s -split-input-file -test-bit-width-constrained-vector-linearize=target-vector-bitwidth=128 | FileCheck %s --check-prefixes=ALL,BW-128
// RUN: mlir-opt %s -split-input-file -test-bit-width-constrained-vector-linearize=target-vector-bitwidth=0   | FileCheck %s --check-prefixes=ALL,BW-0

// A vector<2x2xf32> has inner-most dimension with 64-bits. Check that at
// bitwidth threshold 128 (>= 64), operations are linearized, and at
// bitwidth threshold 0 (< 64), operations are not linearized.

// ALL-LABEL: test_result_bitwidth_64
func.func @test_result_bitwidth_64(%arg0: vector<2x2xf32>) -> vector<2x2xf32> {

  // BW-128:   arith.constant {{.*}} vector<4xf32>
  // BW-0:     arith.constant {{.*}}  vector<2x2xf32>
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>

  // BW-128: math.sin {{.*}} vector<4xf32>
  // BW-0:  math.sin {{.*}}  vector<2x2xf32>
  %1 = math.sin %arg0 : vector<2x2xf32>

  return %0 : vector<2x2xf32>
}

// -----

// The size of the 'index' type is backend specific, so we cannot guarantee that
// the inner-most dimension below (of size 2*nbBits(index)) is below any bitwidth
// threshold. Test that operations with vectors of index type are not linearized.

// ALL-LABEL: test_index_no_linearize
func.func @test_index_no_linearize(%arg0: vector<2x2xindex>, %arg1: vector<2x2xindex>) -> vector<2x2xindex> {

    // BW-128: %[[ADD:.*]] = arith.addi {{.*}} : vector<2x2xindex>
    // BW-0:   %[[ADD:.*]] = arith.addi {{.*}} : vector<2x2xindex>
    %0 = arith.addi %arg0, %arg1 : vector<2x2xindex>
    return %0 : vector<2x2xindex>
}

// -----

// The logic for the insert op with regards to the bitwidth threshold is
// different to the other ops, so we test it here. Specifically, the logic
// is based on the bitwidth of the value to store.

// ALL-LABEL: test_vector_insert
// ALL-SAME: (%[[DEST:.*]]: vector<2x8x4xf32>, %[[SRC:.*]]: vector<8x4xf32>) -> vector<2x8x4xf32> {
func.func @test_vector_insert(%arg0: vector<2x8x4xf32>, %arg1: vector<8x4xf32>) -> vector<2x8x4xf32> {

  // BW-128-DAG: %[[ARG_SRC:.*]] = vector.shape_cast %[[SRC]] : vector<8x4xf32> to vector<32xf32>
  // BW-128-DAG: %[[ARG_DEST:.*]] = vector.shape_cast %[[DEST]] : vector<2x8x4xf32> to vector<64xf32>
  // BW-128: %[[SHUFFLE:.*]] = vector.shuffle %[[ARG_DEST]], %[[ARG_SRC]]
  // BW-128: %[[RES:.*]] = vector.shape_cast %[[SHUFFLE]] : vector<64xf32> to vector<2x8x4xf32>
  // BW-128: return %[[RES]] : vector<2x8x4xf32>

  // BW-0: %[[RES:.*]] = vector.insert %[[SRC]], %[[DEST]] [0] : vector<8x4xf32> into vector<2x8x4xf32>
  // BW-0: return %[[RES]] : vector<2x8x4xf32>

  %0 = vector.insert %arg1, %arg0[0]: vector<8x4xf32> into vector<2x8x4xf32>
  return %0 : vector<2x8x4xf32>
}
