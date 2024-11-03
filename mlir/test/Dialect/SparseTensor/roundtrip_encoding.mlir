// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: func private @sparse_1d_tensor(
// CHECK-SAME: tensor<32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>)
func.func private @sparse_1d_tensor(tensor<32xf64, #sparse_tensor.encoding<{ dimLevelType = ["compressed"] }>>)

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

// CHECK-LABEL: func private @sparse_csr(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 64, indexBitWidth = 64 }>>)
func.func private @sparse_csr(tensor<?x?xf32, #CSR>)

// -----

#CSC = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 0,
  indexBitWidth = 0
}>

// CHECK-LABEL: func private @sparse_csc(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>>)
func.func private @sparse_csc(tensor<?x?xf32, #CSC>)

// -----

#DCSC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 0,
  indexBitWidth = 64
}>

// CHECK-LABEL: func private @sparse_dcsc(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, indexBitWidth = 64 }>>)
func.func private @sparse_dcsc(tensor<?x?xf32, #DCSC>)

// -----

#COO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu-no", "singleton-no" ]
}>

// CHECK-LABEL: func private @sparse_coo(
// CHECK-SAME: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu-no", "singleton-no" ] }>>)
func.func private @sparse_coo(tensor<?x?xf32, #COO>)

// -----

#SortedCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

// CHECK-LABEL: func private @sparse_sorted_coo(
// CHECK-SAME: tensor<10x10xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>>)
func.func private @sparse_sorted_coo(tensor<10x10xf64, #SortedCOO>)

// -----

#BCSR = #sparse_tensor.encoding<{
   dimLevelType = [ "compressed", "compressed", "dense", "dense" ],
   dimOrdering  = affine_map<(ii, jj, i, j) -> (ii, jj, i, j)>,
   higherOrdering = affine_map<(i, j) -> (i floordiv 2, j floordiv 3, i mod 2, j mod 3)>
}>

// CHECK-LABEL: func private @sparse_bcsr(
// CHECK-SAME: tensor<10x60xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "dense", "dense" ], higherOrdering = affine_map<(d0, d1) -> (d0 floordiv 2, d1 floordiv 3, d0 mod 2, d1 mod 3)> }>>
func.func private @sparse_bcsr(tensor<10x60xf64, #BCSR>)


// -----

#ELL = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense", "compressed" ],
  dimOrdering  = affine_map<(ii, i, j) -> (ii, i, j)>,
  higherOrdering = affine_map<(i,j)[c] -> (c*4*i, i, j)>
}>

// CHECK-LABEL: func private @sparse_ell(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense", "compressed" ], higherOrdering = affine_map<(d0, d1)[s0] -> (d0 * (s0 * 4), d0, d1)> }>>
func.func private @sparse_ell(tensor<?x?xf64, #ELL>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (1, 4, 1), (1, 4, 2) ]
}>

// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], slice = [ (1, 4, 1), (1, 4, 2) ] }>>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (1, 4, 1), (1, 4, 2) ]
}>

// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], slice = [ (1, 4, 1), (1, 4, 2) ] }>>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  slice = [ (1, ?, 1), (?, 4, 2) ]
}>

// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], slice = [ (1, ?, 1), (?, 4, 2) ] }>>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)
