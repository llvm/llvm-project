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
     transform.structured.vectorize %0 vector_sizes [1, 4] vectorize_nd_extract : !transform.any_op
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
     transform.structured.vectorize %0 vector_sizes [1, [4]] vectorize_nd_extract : !transform.any_op
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
     transform.structured.vectorize %0 vector_sizes [1, 4] vectorize_nd_extract : !transform.any_op
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
     transform.structured.vectorize %0 vector_sizes [1, [4]] vectorize_nd_extract : !transform.any_op
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

// CHECK-LABEL: func.func @masked_vectorize_nd_tensor_extract_with_affine_apply_gather(
// CHECK-SAME: %[[SRC:.*]]: tensor<80x16xf32>,
// CHECK: %[[MASK:.*]] = vector.create_mask {{.*}} : vector<1x4xi1>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SRC]] {{\[\[}}0, 1]] : tensor<80x16xf32> into tensor<1280xf32>
// CHECK: vector.gather %[[COLLAPSED]][%{{.*}}] [%{{.*}}], %[[MASK]], %{{.*}} : tensor<1280xf32>, vector<1x4xindex>, vector<1x4xi1>, vector<1x4xf32> into vector<1x4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 4] vectorize_nd_extract : !transform.any_op
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

// CHECK-LABEL: func.func @masked_dynamic_vectorize_nd_tensor_extract_with_affine_apply_gather(
// CHECK-SAME: %[[SRC:.*]]: tensor<?x?xf32>,
// CHECK: %[[MASK:.*]] = vector.create_mask {{.*}} : vector<1x4xi1>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SRC]] {{\[\[}}0, 1]] : tensor<?x?xf32> into tensor<?xf32>
// CHECK: vector.gather %[[COLLAPSED]][%{{.*}}] [%{{.*}}], %[[MASK]], %{{.*}} : tensor<?xf32>, vector<1x4xindex>, vector<1x4xi1>, vector<1x4xf32> into vector<1x4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 4] vectorize_nd_extract : !transform.any_op
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

// CHECK-LABEL: func.func @extract_masked_vectorize(
// CHECK-SAME: %[[SRC:.*]]: tensor<?x?xf32>,
// CHECK: %[[MASK:.*]] = vector.create_mask {{.*}} : vector<3x3xi1>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SRC]] {{\[\[}}0, 1]] : tensor<?x?xf32> into tensor<?xf32>
// CHECK: vector.gather %[[COLLAPSED]][%{{.*}}] [%{{.*}}], %[[MASK]], %{{.*}} : tensor<?xf32>, vector<3x3xindex>, vector<3x3xi1>, vector<3x3xf32> into vector<3x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [3, 3] vectorize_nd_extract : !transform.any_op
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
// CHECK-LABEL: func.func @tensor_extract_dynamic_shape(
// CHECK-SAME: %[[SRC:.*]]: tensor<123x321xf32>,
// CHECK: %[[MASK:.*]] = vector.create_mask {{.*}} : vector<1x3x8xi1>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SRC]] {{\[\[}}0, 1]] : tensor<123x321xf32> into tensor<39483xf32>
// CHECK: vector.gather %[[COLLAPSED]][%{{.*}}] [%{{.*}}], %[[MASK]], %{{.*}} : tensor<39483xf32>, vector<1x3x8xindex>, vector<1x3x8xi1>, vector<1x3x8xf32> into vector<1x3x8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 3, 8] vectorize_nd_extract : !transform.any_op
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
// CHECK:           %[[PAD:.*]] = ub.poison : i32
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
    transform.structured.vectorize %0 vector_sizes [1, 1, 4] vectorize_nd_extract : !transform.any_op
    transform.yield
  }
}

// -----

// A contiguous `tensor.extract` from a source whose rank is *smaller* than the
// iteration space (here: a 1-D source inside a 2-D loop nest). The contiguous
// `vector.transfer_read` only reads the trailing (contiguous) dim - the leading
// dim is broadcast via the permutation map - so its mask must be rank-reduced
// (`vector<4xi1>`) rather than the full iteration-space mask (`vector<1x4xi1>`).
// Regression test: previously the full iteration mask was attached, producing
// an invalid `vector.transfer_read` ("inferred mask type ... don't match").

func.func @masked_contiguous_extract_rank_reducing_mask(
    %src: tensor<16xf32>,
    %output : tensor<1x3xf32>,
    %idx: index) -> tensor<1x3xf32> {
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%output : tensor<1x3xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %idx)
    %extracted = tensor.extract %src[%3] : tensor<16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// CHECK-LABEL:   func.func @masked_contiguous_extract_rank_reducing_mask
// CHECK-SAME:      %[[SRC:.*]]: tensor<16xf32>,
// CHECK-SAME:      %[[OUTPUT:.*]]: tensor<1x3xf32>,
// CHECK-SAME:      %[[IDX_IN:.*]]: index) -> tensor<1x3xf32> {

/// Full iteration-space mask (used for the output read/write).
// CHECK:           %[[MASK_2D:.*]] = vector.create_mask {{.*}} : vector<1x4xi1>

/// The contiguous read of the 1-D source uses a rank-reduced mask.
// CHECK:           %[[MASK_1D:.*]] = vector.create_mask {{.*}} : vector<4xi1>
// CHECK:           vector.mask %[[MASK_1D]] {
// CHECK-SAME:          vector.transfer_read %[[SRC]]{{.*}} {in_bounds = [true, true], permutation_map = {{.*}}} : tensor<16xf32>, vector<1x4xf32>
// CHECK-SAME:      } : vector<4xi1> -> vector<1x4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     transform.structured.vectorize %0 vector_sizes [1, 4] vectorize_nd_extract : !transform.any_op
     transform.yield
   }
}

// -----

// Only the inactive fourth lane is out of bounds: the linear offsets are
// [1, 3, 5, 7], while the collapsed source contains six elements.
func.func @masked_gather_inactive_lane_out_of_bounds(
    %src: tensor<3x2xf32>, %out: tensor<3xf32>) -> tensor<3xf32> {
  %c1 = arith.constant 1 : index
  %result = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  } outs(%out : tensor<3xf32>) {
  ^bb0(%unused: f32):
    %i = linalg.index 0 : index
    %value = tensor.extract %src[%i, %c1] : tensor<3x2xf32>
    linalg.yield %value : f32
  } -> tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: func.func @masked_gather_inactive_lane_out_of_bounds(
// CHECK-SAME: %[[SRC:.*]]: tensor<3x2xf32>,
// CHECK: %[[MASK:.*]] = vector.create_mask {{.*}} : vector<4xi1>
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[SRC]] {{\[\[}}0, 1]] : tensor<3x2xf32> into tensor<6xf32>
// CHECK: vector.gather %[[COLLAPSED]][%{{.*}}] [%{{.*}}], %[[MASK]], %{{.*}} : tensor<6xf32>, vector<4xindex>, vector<4xi1>, vector<4xf32> into vector<4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %target = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %target vector_sizes [4] vectorize_nd_extract
      : !transform.any_op
    transform.yield
  }
}

// -----

// Each source dimension fits in int64, but the flattened extent does not.
func.func @negative_gather_from_overflowing_source(
    %src: tensor<3037000500x3037000500xf32>, %out: tensor<2xf32>) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %result = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  } outs(%out : tensor<2xf32>) {
  ^bb0(%unused: f32):
    %i = linalg.index 0 : index
    %value = tensor.extract %src[%i, %c0] : tensor<3037000500x3037000500xf32>
    linalg.yield %value : f32
  } -> tensor<2xf32>
  return %result : tensor<2xf32>
}

// CHECK-LABEL: func.func @negative_gather_from_overflowing_source(
// CHECK-SAME: %[[SRC:.*]]: tensor<3037000500x3037000500xf32>, %[[OUT:.*]]: tensor<2xf32>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[RESULT:.*]] = linalg.generic
// CHECK-NEXT: ^bb0(%{{.*}}: f32):
// CHECK-NEXT: %[[I:.*]] = linalg.index 0 : index
// CHECK-NEXT: %[[VALUE:.*]] = tensor.extract %[[SRC]][%[[I]], %[[ZERO]]]
// CHECK-NEXT: linalg.yield %[[VALUE]] : f32
// CHECK-NEXT: } -> tensor<2xf32>
// CHECK-NEXT: return %[[RESULT]] : tensor<2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // Suppress the expected vectorization failure to check unchanged payload IR.
    transform.sequence %root : !transform.any_op failures(suppress) {
    ^bb0(%scope: !transform.any_op):
      %target = transform.structured.match ops{["linalg.generic"]} in %scope
        : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %target vectorize_nd_extract : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}
