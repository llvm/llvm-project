// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @invalid_new_dense(%arg0: !llvm.ptr<i8>) -> tensor<32xf32> {
  // expected-error@+1 {{expected a sparse tensor result}}
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

func @invalid_release_dense(%arg0: tensor<4xi32>) {
  // expected-error@+1 {{expected a sparse tensor to release}}
  sparse_tensor.release %arg0 : tensor<4xi32>
  return
}

// -----

func @invalid_init_dense(%arg0: index, %arg1: index) -> tensor<?x?xf32> {
  // expected-error@+1 {{expected a sparse tensor result}}
  %0 = sparse_tensor.init [%arg0, %arg1] : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func @invalid_init_rank(%arg0: index) -> tensor<?xf32, #SparseVector> {
  // expected-error@+1 {{unexpected mismatch between tensor rank and sizes: 1 vs. 2}}
  %0 = sparse_tensor.init [%arg0, %arg0] : tensor<?xf32, #SparseVector>
  return %0 : tensor<?xf32, #SparseVector>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

func @invalid_init_size() -> tensor<?x10xf32, #SparseMatrix> {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  // expected-error@+1 {{unexpected mismatch with static dimension size 10}}
  %0 = sparse_tensor.init [%c10, %c20] : tensor<?x10xf32, #SparseMatrix>
  return %0 : tensor<?x10xf32, #SparseMatrix>
}

// -----

func @invalid_pointers_dense(%arg0: tensor<128xf64>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  // expected-error@+1 {{expected a sparse tensor to get pointers}}
  %0 = sparse_tensor.pointers %arg0, %c : tensor<128xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func @invalid_pointers_unranked(%arg0: tensor<*xf64>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  // expected-error@+1 {{expected a sparse tensor to get pointers}}
  %0 = sparse_tensor.pointers %arg0, %c : tensor<*xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"], pointerBitWidth=32}>

func @mismatch_pointers_types(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  // expected-error@+1 {{unexpected type for pointers}}
  %0 = sparse_tensor.pointers %arg0, %c : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func @pointers_oob(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %c = arith.constant 1 : index
  // expected-error@+1 {{requested pointers dimension out of bounds}}
  %0 = sparse_tensor.pointers %arg0, %c : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func @invalid_indices_dense(%arg0: tensor<10x10xi32>) -> memref<?xindex> {
  %c = arith.constant 1 : index
  // expected-error@+1 {{expected a sparse tensor to get indices}}
  %0 = sparse_tensor.indices %arg0, %c : tensor<10x10xi32> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func @invalid_indices_unranked(%arg0: tensor<*xf64>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  // expected-error@+1 {{expected a sparse tensor to get indices}}
  %0 = sparse_tensor.indices %arg0, %c : tensor<*xf64> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func @mismatch_indices_types(%arg0: tensor<?xf64, #SparseVector>) -> memref<?xi32> {
  %c = arith.constant 0 : index
  // expected-error@+1 {{unexpected type for indices}}
  %0 = sparse_tensor.indices %arg0, %c : tensor<?xf64, #SparseVector> to memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func @indices_oob(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %c = arith.constant 1 : index
  // expected-error@+1 {{requested indices dimension out of bounds}}
  %0 = sparse_tensor.indices %arg0, %c : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

func @invalid_values_dense(%arg0: tensor<1024xf32>) -> memref<?xf32> {
  // expected-error@+1 {{expected a sparse tensor to get values}}
  %0 = sparse_tensor.values %arg0 : tensor<1024xf32> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

func @mismatch_values_types(%arg0: tensor<?xf64, #SparseVector>) -> memref<?xf32> {
  // expected-error@+1 {{unexpected mismatch in element types}}
  %0 = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

func @sparse_to_unannotated_tensor(%arg0: memref<?xf64>) -> tensor<16x32xf64> {
  // expected-error@+1 {{expected a sparse tensor result}}
  %0 = sparse_tensor.tensor %arg0 : memref<?xf64> to tensor<16x32xf64>
  return %0 : tensor<16x32xf64>
}

// -----

func @sparse_convert_unranked(%arg0: tensor<*xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{unexpected type in convert}}
  %0 = sparse_tensor.convert %arg0 : tensor<*xf32> to tensor<10xf32>
  return %0 : tensor<10xf32>
}

// -----

#DCSR = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

func @sparse_convert_rank_mismatch(%arg0: tensor<10x10xf64, #DCSR>) -> tensor<?xf64> {
  // expected-error@+1 {{unexpected conversion mismatch in rank}}
  %0 = sparse_tensor.convert %arg0 : tensor<10x10xf64, #DCSR> to tensor<?xf64>
  return %0 : tensor<?xf64>
}

// -----

#CSR = #sparse_tensor.encoding<{dimLevelType = ["dense", "compressed"]}>

func @sparse_convert_dim_mismatch(%arg0: tensor<10x?xf32>) -> tensor<10x10xf32, #CSR> {
  // expected-error@+1 {{unexpected conversion mismatch in dimension 1}}
  %0 = sparse_tensor.convert %arg0 : tensor<10x?xf32> to tensor<10x10xf32, #CSR>
  return %0 : tensor<10x10xf32, #CSR>
}
