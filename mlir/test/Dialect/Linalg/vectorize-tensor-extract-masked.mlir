// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

func.func @masked_static_vectorize_nd_tensor_extract_with_affine_apply_contiguous(
    %src: tensor<80x16xf32>,
    %output : tensor<1x3xf32>,
    %idx: index) -> tensor<1x3xf32> {

  %c79 = arith.constant 79 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%output : tensor<1x3xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %idx)
    %extracted = tensor.extract %src[%c79, %3] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// CHECK-LABEL:   func.func @masked_static_vectorize_nd_tensor_extract_with_affine_apply_contiguous
// CHECK-SAME:      %[[SRC:.*]]: tensor<80x16xf32>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<1x3xf32>,
// CHECK-SAME:      %[[IDX_IN:.*]]: index) -> tensor<1x3xf32> {

/// Create the mask
// CHECK-DAG:       %[[DIM_0:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[DIM_1:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[C79:.*]] = arith.constant 79 : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[DIM_0]], %[[DIM_1]] : vector<1x4xi1>

/// TODO: This transfer_read is redundant - remove
// CHECK:           vector.mask %[[MASK]] { vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<1x3xf32>, vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>

/// Caluclate the index vector
// CHECK:           %[[STEP:.*]] = vector.step : vector<4xindex>
// CHECK:           %[[IDX_BC:.*]] = vector.broadcast %[[IDX_IN]] : index to vector<4xindex>
// CHECK:           %[[IDX_VEC:.*]] = arith.addi %[[STEP]], %[[IDX_BC]] : vector<4xindex>
// CHECK:           %[[SC:.*]] = vector.shape_cast %[[IDX_VEC]] : vector<4xindex> to vector<4xindex>

/// Extract the starting point from the index vector
// CHECK:           %[[IDX_START:.*]] = vector.extract %[[SC]][0] : index from vector<4xindex>

// Final read and write
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC]]{{\[}}%[[C79]], %[[IDX_START]]], {{.*}} {in_bounds = [true, true]} : tensor<80x16xf32>, vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>
// CHECK:           %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:           vector.mask %[[MASK]] { vector.transfer_write %[[READ]], %[[OUTPUT]]{{\[}}%[[C0_1]], %[[C0_1]]] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x3xf32> } : vector<1x4xi1> -> tensor<1x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 4] {vectorize_nd_extract} : !transform.any_op
     transform.yield
   }
}

// -----

// Identical to the above, but with scalable vectors.

func.func @masked_static_vectorize_nd_tensor_extract_with_affine_apply_contiguous_scalable(
    %src: tensor<80x16xf32>,
    %output : tensor<1x3xf32>,
    %idx: index) -> tensor<1x3xf32> {

  %c79 = arith.constant 79 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%output : tensor<1x3xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %idx)
    %extracted = tensor.extract %src[%c79, %3] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x3xf32>

  return %1 : tensor<1x3xf32>
}

// CHECK-LABEL:   func.func @masked_static_vectorize_nd_tensor_extract_with_affine_apply_contiguous_scalable
// CHECK-SAME:      %[[SRC:.*]]: tensor<80x16xf32>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<1x3xf32>,
// CHECK-SAME:      %[[IDX_IN:.*]]: index) -> tensor<1x3xf32> {

/// Create the mask
// CHECK-DAG:       %[[DIM_0:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[DIM_1:.*]] = arith.constant 3 : index
// CHECK-DAG:       %[[C79:.*]] = arith.constant 79 : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[DIM_0]], %[[DIM_1]] : vector<1x[4]xi1>

/// TODO: This transfer_read is redundant - remove
// CHECK:           vector.mask %[[MASK]] { vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<1x3xf32>, vector<1x[4]xf32> } : vector<1x[4]xi1> -> vector<1x[4]xf32>

/// Caluclate the index vector
// CHECK:           %[[STEP:.*]] = vector.step : vector<[4]xindex>
// CHECK:           %[[IDX_BC:.*]] = vector.broadcast %[[IDX_IN]] : index to vector<[4]xindex>
// CHECK:           %[[IDX_VEC:.*]] = arith.addi %[[STEP]], %[[IDX_BC]] : vector<[4]xindex>
// CHECK:           %[[SC:.*]] = vector.shape_cast %[[IDX_VEC]] : vector<[4]xindex> to vector<[4]xindex>

/// Extract the starting point from the index vector
// CHECK:           %[[IDX_START:.*]] = vector.extract %[[SC]][0] : index from vector<[4]xindex>

// Final read and write
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC]]{{\[}}%[[C79]], %[[IDX_START]]], {{.*}} {in_bounds = [true, true]} : tensor<80x16xf32>, vector<1x[4]xf32> } : vector<1x[4]xi1> -> vector<1x[4]xf32>
// CHECK:           %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:           vector.mask %[[MASK]] { vector.transfer_write %[[READ]], %[[OUTPUT]]{{\[}}%[[C0_1]], %[[C0_1]]] {in_bounds = [true, true]} : vector<1x[4]xf32>, tensor<1x3xf32> } : vector<1x[4]xi1> -> tensor<1x3xf32>


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, [4]] {vectorize_nd_extract} : !transform.any_op
     transform.yield
   }
}

// -----

func.func @masked_dynamic_vectorize_nd_tensor_extract_with_affine_apply_contiguous(
    %src: tensor<?x?xf32>,
    %output : tensor<?x?xf32>,
    %idx: index) -> tensor<?x?xf32> {

  %c79 = arith.constant 79 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%output : tensor<?x?xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %idx)
    %extracted = tensor.extract %src[%c79, %3] : tensor<?x?xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @masked_dynamic_vectorize_nd_tensor_extract_with_affine_apply_contiguous(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[OUTPUT:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index)

/// Create the mask
// CHECK:           %[[C79:.*]] = arith.constant 79 : index
// CHECK:           %[[DIM_0_IDX:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM_0:.*]] = tensor.dim %[[OUTPUT]], %[[DIM_0_IDX]] : tensor<?x?xf32>
// CHECK:           %[[DIM_1_IDX:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_1:.*]] = tensor.dim %[[OUTPUT]], %[[DIM_1_IDX]] : tensor<?x?xf32>
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[DIM_0]], %[[DIM_1]] : vector<1x4xi1>

/// TODO: This transfer_read is redundant - remove
// CHECK:           vector.mask %[[MASK]] { vector.transfer_read %[[OUTPUT]]{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>

/// Caluclate the index vector
// CHECK:           %[[STEP:.*]] = vector.step : vector<4xindex>
// CHECK:           %[[IDX_BC:.*]] = vector.broadcast %[[IDX]] : index to vector<4xindex>
// CHECK:           %[[IDX_VEC:.*]] = arith.addi %[[STEP]], %[[IDX_BC]] : vector<4xindex>
// CHECK:           %[[SC:.*]] = vector.shape_cast %[[IDX_VEC]] : vector<4xindex> to vector<4xindex>

/// Extract the starting point from the index vector
// CHECK:           %[[IDX_START:.*]] = vector.extract %[[SC]][0] : index from vector<4xindex>

// Final read and write
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC]]{{\[}}%[[C79]], %[[IDX_START]]], {{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>
// CHECK:           %[[VAL_24:.*]] = vector.mask %[[MASK]] { vector.transfer_write %[[READ]], %[[OUTPUT]]{{.*}} {in_bounds = [true, true]} : vector<1x4xf32>, tensor<?x?xf32> } : vector<1x4xi1> -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 4] {vectorize_nd_extract} : !transform.any_op
     transform.yield
  }
}

// -----

func.func @masked_dynamic_vectorize_nd_tensor_extract_with_affine_apply_contiguous_scalable(
    %src: tensor<?x?xf32>,
    %output : tensor<?x?xf32>,
    %idx: index) -> tensor<?x?xf32> {

  %c79 = arith.constant 79 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%output : tensor<?x?xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %idx)
    %extracted = tensor.extract %src[%c79, %3] : tensor<?x?xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @masked_dynamic_vectorize_nd_tensor_extract_with_affine_apply_contiguous_scalable(
// CHECK-SAME:      %[[SRC:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[OUTPUT:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[IDX:.*]]: index)

/// Create the mask
// CHECK:           %[[C79:.*]] = arith.constant 79 : index
// CHECK:           %[[DIM_0_IDX:.*]] = arith.constant 0 : index
// CHECK:           %[[DIM_0:.*]] = tensor.dim %[[OUTPUT]], %[[DIM_0_IDX]] : tensor<?x?xf32>
// CHECK:           %[[DIM_1_IDX:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM_1:.*]] = tensor.dim %[[OUTPUT]], %[[DIM_1_IDX]] : tensor<?x?xf32>
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[DIM_0]], %[[DIM_1]] : vector<1x[4]xi1>

/// TODO: This transfer_read is redundant - remove
// CHECK:           vector.mask %[[MASK]] { vector.transfer_read %[[OUTPUT]]{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<1x[4]xf32> } : vector<1x[4]xi1> -> vector<1x[4]xf32>

/// Caluclate the index vector
// CHECK:           %[[STEP:.*]] = vector.step : vector<[4]xindex>
// CHECK:           %[[IDX_BC:.*]] = vector.broadcast %[[IDX]] : index to vector<[4]xindex>
// CHECK:           %[[IDX_VEC:.*]] = arith.addi %[[STEP]], %[[IDX_BC]] : vector<[4]xindex>
// CHECK:           %[[SC:.*]] = vector.shape_cast %[[IDX_VEC]] : vector<[4]xindex> to vector<[4]xindex>

/// Extract the starting point from the index vector
// CHECK:           %[[IDX_START:.*]] = vector.extract %[[SC]][0] : index from vector<[4]xindex>

// Final read and write
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC]]{{\[}}%[[C79]], %[[IDX_START]]], {{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<1x[4]xf32> } : vector<1x[4]xi1> -> vector<1x[4]xf32>
// CHECK:           %[[VAL_24:.*]] = vector.mask %[[MASK]] { vector.transfer_write %[[READ]], %[[OUTPUT]]{{.*}} {in_bounds = [true, true]} : vector<1x[4]xf32>, tensor<?x?xf32> } : vector<1x[4]xi1> -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, [4]] {vectorize_nd_extract} : !transform.any_op
     transform.yield
  }
}

// -----

func.func @masked_vectorize_nd_tensor_extract_with_affine_apply_gather(%6: tensor<80x16xf32>, %arg0: index, %extracted_slice : tensor<1x3xf32>) -> tensor<1x3xf32> {
  %c16 = arith.constant 16 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<1x3xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %arg0)
    %extracted = tensor.extract %6[%3, %c16] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// CHECK-LABEL:   func.func @masked_vectorize_nd_tensor_extract_with_affine_apply_gather
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_5]] : vector<1x4xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<1x3xf32>, vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>
// CHECK:           %[[VAL_11:.*]] = vector.broadcast {{.*}} : index to vector<4xindex>
// CHECK:           %[[VAL_12:.*]] = arith.addi {{.*}} : vector<4xindex>
// CHECK:           %[[VAL_16:.*]] = vector.broadcast {{.*}} : vector<4xindex> to vector<1x4xindex>
// CHECK:           %[[VAL_18:.*]] = tensor.dim {{.*}} : tensor<80x16xf32>
// CHECK:           %[[VAL_19:.*]] = vector.broadcast {{.*}} : index to vector<1x4xindex>
// CHECK:           %[[VAL_20:.*]] = arith.muli {{.*}} : vector<1x4xindex>
// CHECK:           %[[VAL_22:.*]] = arith.addi {{.*}} : vector<1x4xindex>
// CHECK:           %[[VAL_23:.*]] = vector.mask %[[VAL_8]] { vector.gather {{.*}} : tensor<80x16xf32>, vector<1x4xindex>, vector<1x4xi1>, vector<1x4xf32> into vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>
// CHECK:           %[[VAL_25:.*]] = vector.mask %[[VAL_8]] { vector.transfer_write {{.*}} {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x3xf32> } : vector<1x4xi1> -> tensor<1x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 4] {vectorize_nd_extract} : !transform.any_op
     transform.yield
   }
}

 // -----

func.func @masked_dynamic_vectorize_nd_tensor_extract_with_affine_apply_gather(%6: tensor<?x?xf32>, %arg0: index, %extracted_slice : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c16 = arith.constant 16 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<?x?xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %arg0)
    %extracted = tensor.extract %6[%3, %c16] : tensor<?x?xf32>
    linalg.yield %extracted : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @masked_dynamic_vectorize_nd_tensor_extract_with_affine_apply_gather(
// CHECK-SAME:                                                                                   %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                                                                                   %[[VAL_1:.*]]: index,
// CHECK-SAME:                                                                                   %[[VAL_2:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_2]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_2]], %[[VAL_6]] : tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = vector.create_mask %[[VAL_5]], %[[VAL_7]] : vector<1x4xi1>
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_10]] { vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_9]] {in_bounds = [true, true]} : tensor<?x?xf32>, vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.step : vector<4xindex>
// CHECK:           %[[VAL_13:.*]] = vector.broadcast %[[VAL_1]] : index to vector<4xindex>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : vector<4xindex>
// CHECK:           %[[VAL_15:.*]] = arith.constant dense<true> : vector<1x4xi1>
// CHECK:           %[[VAL_16:.*]] = arith.constant dense<0.000000e+00> : vector<1x4xf32>
// CHECK:           %[[VAL_17:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_18:.*]] = vector.broadcast %[[VAL_14]] : vector<4xindex> to vector<1x4xindex>
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = tensor.dim %[[VAL_0]], %[[VAL_19]] : tensor<?x?xf32>
// CHECK:           %[[VAL_21:.*]] = vector.broadcast %[[VAL_20]] : index to vector<1x4xindex>
// CHECK:           %[[VAL_22:.*]] = arith.muli %[[VAL_18]], %[[VAL_21]] : vector<1x4xindex>
// CHECK:           %[[VAL_23:.*]] = arith.constant dense<16> : vector<1x4xindex>
// CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_22]] : vector<1x4xindex>
// CHECK:           %[[VAL_25:.*]] = vector.mask %[[VAL_10]] { vector.gather %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_17]]] {{\[}}%[[VAL_24]]], %[[VAL_15]], %[[VAL_16]] : tensor<?x?xf32>, vector<1x4xindex>, vector<1x4xi1>, vector<1x4xf32> into vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>
// CHECK:           %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_27:.*]] = vector.mask %[[VAL_10]] { vector.transfer_write %[[VAL_25]], %[[VAL_2]]{{\[}}%[[VAL_26]], %[[VAL_26]]] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<?x?xf32> } : vector<1x4xi1> -> tensor<?x?xf32>
// CHECK:           return %[[VAL_27]] : tensor<?x?xf32>
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 4] {vectorize_nd_extract} : !transform.any_op
     transform.yield
   }
}

// -----

#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @extract_masked_vectorize(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 2 : index
  %2 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel"]
  } outs(%arg1 : tensor<?x?xf32>) {
  ^bb0(%arg3: f32):
    %7 = tensor.extract %arg0[%c0, %c1] : tensor<?x?xf32>
    linalg.yield %7 : f32
  } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL:   func.func @extract_masked_vectorize(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[VAL_6]] : tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = vector.create_mask %[[VAL_5]], %[[VAL_7]] : vector<3x3xi1>
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_10]] { vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_9]] {in_bounds = [true, true]} : tensor<?x?xf32>, vector<3x3xf32> } : vector<3x3xi1> -> vector<3x3xf32>
// CHECK:           %[[VAL_12:.*]] = arith.constant dense<true> : vector<3x3xi1>
// CHECK:           %[[VAL_13:.*]] = arith.constant dense<0.000000e+00> : vector<3x3xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant dense<1> : vector<3x3xindex>
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_17:.*]] = tensor.dim %[[VAL_0]], %[[VAL_16]] : tensor<?x?xf32>
// CHECK:           %[[VAL_18:.*]] = vector.broadcast %[[VAL_17]] : index to vector<3x3xindex>
// CHECK:           %[[VAL_19:.*]] = arith.muli %[[VAL_15]], %[[VAL_18]] : vector<3x3xindex>
// CHECK:           %[[VAL_20:.*]] = arith.constant dense<2> : vector<3x3xindex>
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_19]] : vector<3x3xindex>
// CHECK:           %[[VAL_22:.*]] = vector.mask %[[VAL_10]] { vector.gather %[[VAL_0]]{{\[}}%[[VAL_14]], %[[VAL_14]]] {{\[}}%[[VAL_21]]], %[[VAL_12]], %[[VAL_13]] : tensor<?x?xf32>, vector<3x3xindex>, vector<3x3xi1>, vector<3x3xf32> into vector<3x3xf32> } : vector<3x3xi1> -> vector<3x3xf32>
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = vector.mask %[[VAL_10]] { vector.transfer_write %[[VAL_22]], %[[VAL_1]]{{\[}}%[[VAL_23]], %[[VAL_23]]] {in_bounds = [true, true]} : vector<3x3xf32>, tensor<?x?xf32> } : vector<3x3xi1> -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [3, 3] {vectorize_nd_extract} : !transform.any_op
     transform.yield
   }
}

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @tensor_extract_dynamic_shape(%arg1: tensor<123x321xf32>, %arg2: tensor<1x?x8xf32>) -> tensor<1x?x8xf32> {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 2 : index
  %2 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x?x8xf32>)
  {
  ^bb0(%arg3: f32):
    %idx_0 = linalg.index 0 : index
    %idx_1 = linalg.index 1 : index
    %idx = arith.addi %idx_0, %idx_1 : index
    %7 = tensor.extract %arg1[%c0, %idx] : tensor<123x321xf32>
    linalg.yield %7 : f32
  } -> tensor<1x?x8xf32>
  return %2 : tensor<1x?x8xf32>
} 

// TODO: Make sure that this is vectorized as "scalar broadcast" when only
// vectorising the 2nd dimension.
// CHECK-LABEL:   func.func @tensor_extract_dynamic_shape(
// CHECK-SAME:      %[[ARG_1:.*]]: tensor<123x321xf32>,
// CHECK-SAME:      %[[ARG_2:.*]]: tensor<1x?x8xf32>) -> tensor<1x?x8xf32> {
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[C1_1:.*]] = arith.constant 1 : index
// CHECK:           %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:           %[[DIM:.*]] = tensor.dim %[[ARG_2]], %[[C1_2]] : tensor<1x?x8xf32>
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[C1_1]], %[[DIM]], %[[C8]] : vector<1x3x8xi1>
// CHECK:           %[[MASK_2:.*]] = arith.constant dense<true> : vector<1x3x8xi1>
// CHECK:           %[[FALLTHROUGH:.*]] = arith.constant dense<0.000000e+00> : vector<1x3x8xf32>
// CHECK:           %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:           vector.mask %[[MASK]] { vector.gather %[[ARG_1]][%[[C0_1]], %[[C0_1]]] [%{{.*}}], %[[MASK_2]], %[[FALLTHROUGH]] : tensor<123x321xf32>, vector<1x3x8xindex>, vector<1x3x8xi1>, vector<1x3x8xf32> into vector<1x3x8xf32> } : vector<1x3x8xi1> -> vector<1x3x8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 3, 8] {vectorize_nd_extract} : !transform.any_op
     transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @scalar_broadcast(%init : tensor<1x1x3xi32>, %src: tensor<1x3x2x4xi32>, %idx :index) -> tensor<1x1x3xi32> {

  %c0 = arith.constant 0 :index

  %res = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel", "parallel"]}
    outs(%init : tensor<1x1x3xi32>) {
    ^bb0(%out: i32):
      %val = tensor.extract %src[%idx, %idx, %idx, %idx] : tensor<1x3x2x4xi32>
      linalg.yield %val : i32
  } -> tensor<1x1x3xi32>

  return %res : tensor<1x1x3xi32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (0, 0, 0)>
// CHECK-LABEL:   func.func @scalar_broadcast(
// CHECK-SAME:      %[[INIT:.*]]: tensor<1x1x3xi32>,
// CHECK-SAME:      %[[SRC:.*]]: tensor<1x3x2x4xi32>,
// CHECK-SAME:      %[[IDX:.*]]: index) -> tensor<1x1x3xi32> {

/// Compute the mask for saving the final result
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C1_2:.*]] = arith.constant 1 : index
// CHECK:           %[[C3:.*]] = arith.constant 3 : index
// CHECK:           %[[MASK_RES:.*]] = vector.create_mask %[[C1]], %[[C1_2]], %[[C3]] : vector<1x1x4xi1>

/// Read and broadcast the scalar
// CHECK:           %[[PAD:.*]] = arith.constant 0 : i32
// CHECK:           %[[MASK_READ:.*]] = vector.constant_mask [1] : vector<1xi1>
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK_READ]] {
// CHECK-SAME:          vector.transfer_read %[[SRC]]{{\[}}%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]],  %[[PAD]]
// CHECK-SAME:          {in_bounds = [true, true, true], permutation_map = #[[$MAP]]} : tensor<1x3x2x4xi32>, vector<1x1x4xi32>
// CHECK-SAME:      } : vector<1xi1> -> vector<1x1x4xi32>

/// Save the result in the output tensor
// CHECK:           vector.mask %[[MASK_RES]] {
// CHECK-SAME:        vector.transfer_write %[[READ]], %[[INIT]]{{.*}} {in_bounds = [true, true, true]} : vector<1x1x4xi32>, tensor<1x1x3xi32>
// CHECK-SAME:      } : vector<1x1x4xi1> -> tensor<1x1x3xi32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [1, 1, 4] {vectorize_nd_extract} : !transform.any_op
    transform.yield
  }
}
