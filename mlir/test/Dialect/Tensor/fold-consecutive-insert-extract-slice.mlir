// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-fold-consecutive-insert-extract-slice -canonicalize -mlir-print-local-scope %s | FileCheck %s

func.func @extract_slice_same_rank(
    %src: tensor<?x?x?x?xf32>, %offset0: index, %offset1: index, %size0: index, %size1: index) -> tensor<8x16x32x?xf32> {
  %0 = tensor.extract_slice %src[0, 1, 2, %offset0] [128, 128, 128, %size0] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<128x128x128x?xf32>
  %1 = tensor.extract_slice %0[7, 8, 9, %offset1] [8, 16, 32, %size1] [1, 1, 1, 1] : tensor<128x128x128x?xf32> to tensor<8x16x32x?xf32>
  return %1: tensor<8x16x32x?xf32>
}

// CHECK-LABEL: func.func @extract_slice_same_rank
//  CHECK-SAME: (%[[SOURCE:.+]]: tensor<?x?x?x?xf32>, %[[OFFSET0:.+]]: index, %[[OFFSET1:.+]]: index, %{{.+}}: index, %[[SIZE1:.+]]: index)
//       CHECK:   %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[OFFSET1]], %[[OFFSET0]]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SOURCE]][7, 9, 11, %[[OFFSET]]] [8, 16, 32, %[[SIZE1]]] [1, 1, 1, 1]
//       CHECK:   return %[[EXTRACT]] : tensor<8x16x32x?xf32>

// -----

func.func @extract_slice_rank_reducing_consumer(
    %src: tensor<?x?x?x?xf32>, %offset0: index, %offset1: index, %size0: index, %size1: index) -> tensor<16x?xf32> {
  %0 = tensor.extract_slice %src[0, 1, 2, %offset0] [128, 128, 128, %size0] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<128x128x128x?xf32>
  %1 = tensor.extract_slice %0[7, 8, 9, %offset1] [1, 16, 1, %size1] [1, 1, 1, 1] : tensor<128x128x128x?xf32> to tensor<16x?xf32>
  return %1: tensor<16x?xf32>
}

// CHECK-LABEL: func.func @extract_slice_rank_reducing_consumer
//       CHECK:   tensor.extract_slice %{{.+}}[7, 9, 11, %{{.+}}] [1, 16, 1, %{{.+}}] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<16x?xf32>

// -----

func.func @extract_slice_rank_reducing_producer(
    %src: tensor<?x?x?x?xf32>, %offset0: index, %offset1: index, %size0: index, %size1: index) -> tensor<8x?xf32> {
  %0 = tensor.extract_slice %src[0, 1, 2, %offset0] [1, 128, 1, %size0] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<128x?xf32>
  %1 = tensor.extract_slice %0[7, %offset1] [8, %size1] [1, 1] : tensor<128x?xf32> to tensor<8x?xf32>
  return %1: tensor<8x?xf32>
}

// CHECK-LABEL: func.func @extract_slice_rank_reducing_producer
//  CHECK-SAME: (%[[SRC:.+]]: tensor<?x?x?x?xf32>, %[[OFFSET0:.+]]: index, %[[OFFSET1:.+]]: index, %{{.+}}: index, %[[SIZE1:.+]]: index)
//       CHECK:   %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%[[OFFSET1]], %[[OFFSET0]]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SRC]][0, 8, 2, %[[OFFSET]]] [1, 8, 1, %[[SIZE1]]] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<8x?xf32>
//       CHECK:   return %[[EXTRACT]] : tensor<8x?xf32>

// -----

func.func @extract_slice_non_one_stride(
    %src: tensor<?xf32>, %offset0: index, %offset1: index, %size0: index, %size1: index, %stride0: index, %stride1: index) -> tensor<?xf32> {
  %0 = tensor.extract_slice %src[%offset0] [%size0] [%stride0] : tensor<?xf32> to tensor<?xf32>
  %1 = tensor.extract_slice %0[%offset1] [%size1] [%stride1] : tensor<?xf32> to tensor<?xf32>
  return %1: tensor<?xf32>
}

// CHECK-LABEL: func.func @extract_slice_non_one_stride
//  CHECK-SAME: (%[[SRC:.+]]: tensor<?xf32>, %[[OFFSET0:.+]]: index, %[[OFFSET1:.+]]: index, %{{.+}}: index, %[[SIZE1:.+]]: index, %[[STRIDE0:.+]]: index, %[[STRIDE1:.+]]: index)
//       CHECK:   %[[OFFSET:.+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>()[%[[OFFSET1]], %[[STRIDE0]], %[[OFFSET0]]]
//       CHECK:   %[[STRIDE:.+]] = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%[[STRIDE1]], %[[STRIDE0]]]
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SRC]][%[[OFFSET]]] [%[[SIZE1]]] [%[[STRIDE]]] : tensor<?xf32> to tensor<?xf32>
//       CHECK:   return %[[EXTRACT]] : tensor<?xf32>

// -----

func.func @insert_slice_rank_reducing(
    %dst: tensor<128x128x128x128xf32>, %mid: tensor<1x16x1xf32>, %src: tensor<16xf32>, %offset: index) -> tensor<128x128x128x128xf32> {
  %0 = tensor.insert_slice %src into %mid[0, 0, 0] [1, 16, 1] [1, 1, 1] : tensor<16xf32> into tensor<1x16x1xf32>
  %1 = tensor.insert_slice %0 into %dst[6, 7, 8, %offset] [1, 1, 16, 1] [1, 1, 1, 1] : tensor<1x16x1xf32> into tensor<128x128x128x128xf32>
  return %1: tensor<128x128x128x128xf32>
}

// CHECK-LABEL: func.func @insert_slice_rank_reducing
//  CHECK-SAME: (%[[DST:.+]]: tensor<128x128x128x128xf32>, %{{.+}}: tensor<1x16x1xf32>, %[[SRC:.+]]: tensor<16xf32>, %[[IDX:.+]]: index)
//       CHECK:  %[[INSERT:.+]] = tensor.insert_slice %[[SRC]] into %[[DST]][6, 7, 8, %[[IDX]]] [1, 1, 16, 1] [1, 1, 1, 1]
//       CHECK:  return %[[INSERT]]

// -----

func.func @insert_slice_rank_reducing_dynamic_shape(
    %dst: tensor<128x128x128x128xf32>, %mid: tensor<1x?x1xf32>, %src: tensor<?xf32>, %offset: index, %size: index) -> tensor<128x128x128x128xf32> {
  %0 = tensor.insert_slice %src into %mid[0, 0, 0] [1, %size, 1] [1, 1, 1] : tensor<?xf32> into tensor<1x?x1xf32>
  %1 = tensor.insert_slice %0 into %dst[6, 7, 8, %offset] [1, 1, %size, 1] [1, 1, 1, 1] : tensor<1x?x1xf32> into tensor<128x128x128x128xf32>
  return %1: tensor<128x128x128x128xf32>
}

//   CHECK-LABEL: func.func @insert_slice_rank_reducing_dynamic_shape
// CHECK-COUNT-2:   tensor.insert_slice
