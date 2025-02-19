// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-eliminate-whole-slicing-patterns -canonicalize -mlir-print-local-scope %s | FileCheck %s

//////////////////////////////
// here starts the tests for insert_slice
//////////////////////////////

func.func @elim_dyn_insert(%arg0: tensor<32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<32x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  %c0f = arith.constant 0.0 : bf16
  %3 = linalg.fill ins(%c0f : bf16) outs(%extracted_slice : tensor<?x32x32x32xbf16>) -> tensor<?x32x32x32xbf16>
  %inserted_slice = tensor.insert_slice %3 into %extracted_slice[0, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<?x32x32x32xbf16> into tensor<?x32x32x32xbf16>
  %inserted_slice_3 = tensor.insert_slice %inserted_slice into %arg0[%arg2, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<?x32x32x32xbf16> into tensor<32x32x32x32xbf16>
  return %inserted_slice_3 : tensor<32x32x32x32xbf16>
}

// CHECK-LABEL: func.func @elim_dyn_insert
//  CHECK-SAME: (%[[SOURCE:.+]]: tensor<32x32x32x32xbf16>, %[[OFFSET0:.+]]: index, %[[OFFSET1:.+]]: index
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SOURCE]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice %[[FILL]] into %[[SOURCE]]
//       CHECK:   return %[[INSERT]]

func.func @elim_static_insert(%arg0: tensor<32x32x32x32xbf16>, %arg2: index) -> tensor<32x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  %c0f = arith.constant 0.0 : bf16
  %3 = linalg.fill ins(%c0f : bf16) outs(%extracted_slice : tensor<15x32x32x32xbf16>) -> tensor<15x32x32x32xbf16>
  %inserted_slice = tensor.insert_slice %3 into %extracted_slice[0, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> into tensor<15x32x32x32xbf16>
  %inserted_slice_3 = tensor.insert_slice %inserted_slice into %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> into tensor<32x32x32x32xbf16>
  return %inserted_slice_3 : tensor<32x32x32x32xbf16>
}

// CHECK-LABEL: func.func @elim_static_insert
//  CHECK-SAME: (%[[SOURCE:.+]]: tensor<32x32x32x32xbf16>, %[[OFFSET0:.+]]: index
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SOURCE]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice %[[FILL]] into %[[SOURCE]]
//       CHECK:   return %[[INSERT]]

func.func @fail_dyn_insert_shape(%arg0: tensor<32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<32x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  %c0f = arith.constant 0.0 : bf16
  %3 = linalg.fill ins(%c0f : bf16) outs(%extracted_slice : tensor<?x32x32x32xbf16>) -> tensor<?x32x32x32xbf16>
  %inserted_slice = tensor.insert_slice %3 into %extracted_slice[0, 0, 0, 0] [%arg2, 32, 32, 32] [1, 1, 1, 1] : tensor<?x32x32x32xbf16> into tensor<?x32x32x32xbf16>
  %inserted_slice_3 = tensor.insert_slice %inserted_slice into %arg0[%arg2, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<?x32x32x32xbf16> into tensor<32x32x32x32xbf16>
  return %inserted_slice_3 : tensor<32x32x32x32xbf16>
}
// fail to optimize due to unmatched insert shape
// CHECK-LABEL: func.func @fail_dyn_insert_shape
//  CHECK-SAME: (%[[SOURCE:.+]]: tensor<32x32x32x32xbf16>, %[[OFFSET0:.+]]: index
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SOURCE]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   tensor.insert_slice
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   return %[[INSERT]]

func.func @fail_static_insert_shape(%arg0: tensor<32x32x32x32xbf16>, %arg2: index) -> tensor<32x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  %3 = tensor.empty() : tensor<14x32x32x32xbf16>
  %inserted_slice = tensor.insert_slice %3 into %extracted_slice[0, 0, 0, 0] [14, 32, 32, 32] [1, 1, 1, 1] : tensor<14x32x32x32xbf16> into tensor<15x32x32x32xbf16>
  %inserted_slice_3 = tensor.insert_slice %inserted_slice into %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> into tensor<32x32x32x32xbf16>
  return %inserted_slice_3 : tensor<32x32x32x32xbf16>
}
// fail to optimize due to unmatched insert shape
// CHECK-LABEL: func.func @fail_static_insert_shape
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:   tensor.empty()
//       CHECK:   tensor.insert_slice
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   return %[[INSERT]]

func.func @fail_dyn_insert_stride(%arg0: tensor<32x32x32x32xbf16>, %arg2: index) -> tensor<32x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  %c0f = arith.constant 0.0 : bf16
  %3 = linalg.fill ins(%c0f : bf16) outs(%extracted_slice : tensor<15x32x32x32xbf16>) -> tensor<15x32x32x32xbf16>
  %inserted_slice = tensor.insert_slice %3 into %extracted_slice[0, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, %arg2] : tensor<15x32x32x32xbf16> into tensor<15x32x32x32xbf16>
  %inserted_slice_3 = tensor.insert_slice %inserted_slice into %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> into tensor<32x32x32x32xbf16>
  return %inserted_slice_3 : tensor<32x32x32x32xbf16>
}
// fail to optimize due to dynamic stride
// CHECK-LABEL: func.func @fail_dyn_insert_stride
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   tensor.insert_slice
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   return %[[INSERT]]

// fail to optimize due to non-zero offset
func.func @fail_static_insert_offset(%arg0: tensor<32x32x32x32xbf16>, %arg2: index) -> tensor<32x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  %c0f = arith.constant 0.0 : bf16
  %3 = linalg.fill ins(%c0f : bf16) outs(%extracted_slice : tensor<15x32x32x32xbf16>) -> tensor<15x32x32x32xbf16>
  %inserted_slice = tensor.insert_slice %3 into %extracted_slice[0, 0, 0, 1] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> into tensor<15x32x32x32xbf16>
  %inserted_slice_3 = tensor.insert_slice %inserted_slice into %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> into tensor<32x32x32x32xbf16>
  return %inserted_slice_3 : tensor<32x32x32x32xbf16>
}
// CHECK-LABEL: func.func @fail_static_insert_offset
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK:   tensor.insert_slice
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   return %[[INSERT]]

//////////////////////////////
// here starts the tests for extract_slice
//////////////////////////////
func.func @elim_dyn_extract(%arg0: tensor<32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<?x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  %extracted_slice2 = tensor.extract_slice %extracted_slice[0, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<?x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  return %extracted_slice2 : tensor<?x32x32x32xbf16>
}
// CHECK-LABEL: func.func @elim_dyn_extract
//  CHECK-SAME: (%[[SOURCE:.+]]: tensor<32x32x32x32xbf16>, %[[OFFSET0:.+]]: index, %[[OFFSET1:.+]]: index
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SOURCE]][%[[OFFSET0]], 0, 0, 0] [%[[OFFSET1]], 32, 32, 32]
//       CHECK:   return %[[EXTRACT]]


func.func @elim_static_extract(%arg0: tensor<32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<15x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  %extracted_slice2 = tensor.extract_slice %extracted_slice[0, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  return %extracted_slice2 : tensor<15x32x32x32xbf16>
}
// CHECK-LABEL: func.func @elim_static_extract
//  CHECK-SAME: (%[[SOURCE:.+]]: tensor<32x32x32x32xbf16>, %[[OFFSET0:.+]]: index, %[[OFFSET1:.+]]: index
//       CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[SOURCE]][%[[OFFSET0]], 0, 0, 0] [15, 32, 32, 32]
//       CHECK:   return %[[EXTRACT]]

// fail to optimize due to unmatched shape
func.func @fail_dyn_extract_shape(%arg0: tensor<32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<?x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  %extracted_slice2 = tensor.extract_slice %extracted_slice[0, 0, 0, 0] [%arg2, 32, 32, 32] [1, 1, 1, 1] : tensor<?x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  return %extracted_slice2 : tensor<?x32x32x32xbf16>
}
// CHECK-LABEL: func.func @fail_dyn_extract_shape
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice
//       CHECK:   return

// fail to optimize due to unmatched shape
func.func @fail_static_extract_shape(%arg0: tensor<32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<14x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  %extracted_slice2 = tensor.extract_slice %extracted_slice[0, 0, 0, 0] [14, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> to tensor<14x32x32x32xbf16>
  return %extracted_slice2 : tensor<14x32x32x32xbf16>
}
// CHECK-LABEL: func.func @fail_static_extract_shape
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice
//       CHECK:   return

// fail to optimize due to stride
func.func @fail_extract_stride(%arg0: tensor<32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<?x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  %extracted_slice2 = tensor.extract_slice %extracted_slice[0, 0, 0, 0] [%arg3, 32, 32, 32] [1, 1, 1, 3] : tensor<?x32x32x32xbf16> to tensor<?x32x32x32xbf16>
  return %extracted_slice2 : tensor<?x32x32x32xbf16>
}
// CHECK-LABEL: func.func @fail_extract_stride
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice
//       CHECK:   return

// fail to optimize due to non-zero offset
func.func @fail_static_extract_offset(%arg0: tensor<32x32x32x32xbf16>, %arg2: index) -> tensor<15x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<32x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  %extracted_slice2 = tensor.extract_slice %extracted_slice[%arg2, 0, 0, 0] [15, 32, 32, 32] [1, 1, 1, 1] : tensor<15x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  return %extracted_slice2 : tensor<15x32x32x32xbf16>
}
// CHECK-LABEL: func.func @fail_static_extract_offset
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice
//       CHECK:   return



//////////////////////////////
// here starts the tests for expanding/reducing dims
//////////////////////////////
func.func @fail_extract_reduce(%arg0: tensor<1x32x32x32x32xbf16>, %arg2: index, %arg3: index) -> tensor<15x32x32x32xbf16> {
  %extracted_slice = tensor.extract_slice %arg0[0, %arg2, 0, 0, 0] [1, 15, 32, 32, 32] [1, 1, 1, 1, 1] : tensor<1x32x32x32x32xbf16> to tensor<1x15x32x32x32xbf16>
  %extracted_slice2 = tensor.extract_slice %extracted_slice[0, 0, 0, 0, 0] [1, 15, 32, 32, 32] [1, 1, 1, 1, 1] : tensor<1x15x32x32x32xbf16> to tensor<15x32x32x32xbf16>
  return %extracted_slice2 : tensor<15x32x32x32xbf16>
}
// CHECK-LABEL: func.func @fail_extract_reduce
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice
//       CHECK:   return

func.func @fail_insert_expand(%arg0: tensor<1x15x32x32x32xbf16>, %arg1: tensor<1x15x32x32x32xbf16>, %arg2: index) -> tensor<1x15x32x32x32xbf16> {
  %extracted_slice = tensor.empty(): tensor<15x32x32x32xbf16>
  %extracted_slice2 = tensor.insert_slice %extracted_slice into %arg0[0, 0, 0, 0, 0] [1, 15, 32, 32, 32] [1, 1, 1, 1, 1] : tensor<15x32x32x32xbf16> into tensor<1x15x32x32x32xbf16>
  return %extracted_slice2 : tensor<1x15x32x32x32xbf16>
}
// CHECK-LABEL: func.func @fail_insert_expand
//       CHECK:   tensor.empty
//       CHECK:   tensor.insert_slice
//       CHECK:   return