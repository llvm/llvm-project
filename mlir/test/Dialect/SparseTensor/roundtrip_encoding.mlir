// RUN: mlir-opt %s -split-input-file | mlir-opt -split-input-file | FileCheck %s

#SV  = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK: #[[$SV:.*]] = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
// CHECK-LABEL: func private @sparse_1d_tensor(
// CHECK-SAME: tensor<32xf64, #[[$SV]]>)
func.func private @sparse_1d_tensor(tensor<32xf64, #SV>)

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 64,
  crdWidth = 64
}>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed), posWidth = 64, crdWidth = 64 }>
// CHECK-LABEL: func private @sparse_csr(
// CHECK-SAME: tensor<?x?xf32, #[[$CSR]]>)
func.func private @sparse_csr(tensor<?x?xf32, #CSR>)

// -----

#CSR_explicit = #sparse_tensor.encoding<{
  map = {l0, l1} (d0 = l0, d1 = l1) -> (l0 = d0 : dense, l1 = d1 : compressed)
}>

// CHECK: #[[$CSR_EXPLICIT:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func private @CSR_explicit(
// CHECK-SAME: tensor<?x?xf64, #[[$CSR_EXPLICIT]]>
func.func private @CSR_explicit(%arg0: tensor<?x?xf64, #CSR_explicit>) {
  return
}

// -----

#CSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : dense, d0 : compressed),
  posWidth = 0,
  crdWidth = 0
}>

// CHECK-DAG: #[[$CSC:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d1 : dense, d0 : compressed) }>
// CHECK-LABEL: func private @sparse_csc(
// CHECK-SAME: tensor<?x?xf32, #[[$CSC]]>)
func.func private @sparse_csc(tensor<?x?xf32, #CSC>)

// -----

#DCSC = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed),
  posWidth = 0,
  crdWidth = 64
}>

// CHECK-DAG: #[[$DCSC:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d1 : compressed, d0 : compressed), crdWidth = 64 }>
// CHECK-LABEL: func private @sparse_dcsc(
// CHECK-SAME: tensor<?x?xf32, #[[$DCSC]]>)
func.func private @sparse_dcsc(tensor<?x?xf32, #DCSC>)

// -----

#COO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique, nonordered), d1 : singleton(nonordered))
}>

// CHECK-DAG: #[[$COO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique, nonordered), d1 : singleton(nonordered)) }>
// CHECK-LABEL: func private @sparse_coo(
// CHECK-SAME: tensor<?x?xf32, #[[$COO]]>)
func.func private @sparse_coo(tensor<?x?xf32, #COO>)

// -----

#BCOO = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : loose_compressed(nonunique), d2 : singleton)
}>

// CHECK-DAG: #[[$BCOO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : loose_compressed(nonunique), d2 : singleton) }>
// CHECK-LABEL: func private @sparse_bcoo(
// CHECK-SAME: tensor<?x?x?xf32, #[[$BCOO]]>)
func.func private @sparse_bcoo(tensor<?x?x?xf32, #BCOO>)

// -----

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

// CHECK-DAG: #[[$SortedCOO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton) }>
// CHECK-LABEL: func private @sparse_sorted_coo(
// CHECK-SAME: tensor<10x10xf64, #[[$SortedCOO]]>)
func.func private @sparse_sorted_coo(tensor<10x10xf64, #SortedCOO>)

// -----

#COO_SoA = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa))
}>

// CHECK-DAG: #[[$COO_SoA:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton(soa)) }>
// CHECK-LABEL: func private @sparse_coo(
// CHECK-SAME: tensor<?x?xf32, #[[$COO_SoA]]>)
func.func private @sparse_coo(tensor<?x?xf32, #COO_SoA>)

// -----

#BSR = #sparse_tensor.encoding<{
   map = ( i, j ) ->
      ( i floordiv 2 : dense,
        j floordiv 3 : compressed,
        i mod 2      : dense,
        j mod 3      : dense
      )
}>

// CHECK-DAG: #[[$BSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 2 : dense, d1 floordiv 3 : compressed, d0 mod 2 : dense, d1 mod 3 : dense) }>
// CHECK-LABEL: func private @sparse_bsr(
// CHECK-SAME: tensor<10x60xf64, #[[$BSR]]>
func.func private @sparse_bsr(tensor<10x60xf64, #BSR>)


// -----

#ELL = #sparse_tensor.encoding<{
  map = [s0](d0, d1) -> (d0 * (s0 * 4) : dense, d0 : dense, d1 : compressed)
}>

// CHECK-DAG: #[[$ELL:.*]] = #sparse_tensor.encoding<{ map = [s0](d0, d1) -> (d0 * (s0 * 4) : dense, d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func private @sparse_ell(
// CHECK-SAME: tensor<?x?xf64, #[[$ELL]]>
func.func private @sparse_ell(tensor<?x?xf64, #ELL>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(1, 4, 1)>, d1 : #sparse_tensor<slice(1, 4, 2)>) -> (d0 : dense, d1 : compressed)
}>

// CHECK-DAG: #[[$CSR_SLICE:.*]] = #sparse_tensor.encoding<{ map = (d0 : #sparse_tensor<slice(1, 4, 1)>, d1 : #sparse_tensor<slice(1, 4, 2)>) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #[[$CSR_SLICE]]>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

#CSR_SLICE = #sparse_tensor.encoding<{
  map = (d0 : #sparse_tensor<slice(1, ?, 1)>, d1 : #sparse_tensor<slice(?, 4, 2)>) -> (d0 : dense, d1 : compressed)
}>

// CHECK-DAG: #[[$CSR_SLICE:.*]] = #sparse_tensor.encoding<{ map = (d0 : #sparse_tensor<slice(1, ?, 1)>, d1 : #sparse_tensor<slice(?, 4, 2)>) -> (d0 : dense, d1 : compressed) }>
// CHECK-LABEL: func private @sparse_slice(
// CHECK-SAME: tensor<?x?xf64, #[[$CSR_SLICE]]>
func.func private @sparse_slice(tensor<?x?xf64, #CSR_SLICE>)

// -----

#BSR = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i floordiv 2 : dense,
    j floordiv 3 : compressed,
    i mod 2      : dense,
    j mod 3      : dense
  )
}>

// CHECK-DAG: #[[$BSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 2 : dense, d1 floordiv 3 : compressed, d0 mod 2 : dense, d1 mod 3 : dense) }>
// CHECK-LABEL: func private @BSR(
// CHECK-SAME: tensor<?x?xf64, #[[$BSR]]>
func.func private @BSR(%arg0: tensor<?x?xf64, #BSR>) {
  return
}

// -----

#BCSR = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i floordiv 2 : dense,
    j floordiv 3 : dense,
    k floordiv 4 : compressed,
    i mod 2      : dense,
    j mod 3      : dense,
    k mod 4      : dense
  )
}>

// CHECK-DAG: #[[$BCSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 floordiv 2 : dense, d1 floordiv 3 : dense, d2 floordiv 4 : compressed, d0 mod 2 : dense, d1 mod 3 : dense, d2 mod 4 : dense) }>
// CHECK-LABEL: func private @BCSR(
// CHECK-SAME: tensor<?x?x?xf64, #[[$BCSR]]>
func.func private @BCSR(%arg0: tensor<?x?x?xf64, #BCSR>) {
  return
}
// -----

#BSR_explicit = #sparse_tensor.encoding<{
  map =
  {il, jl, ii, jj}
  ( i = il * 2 + ii,
    j = jl * 3 + jj
  ) ->
  ( il = i floordiv 2 : dense,
    jl = j floordiv 3 : compressed,
    ii = i mod 2      : dense,
    jj = j mod 3      : dense
  )
}>

// CHECK-DAG: #[[$BSR_explicit:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 2 : dense, d1 floordiv 3 : compressed, d0 mod 2 : dense, d1 mod 3 : dense) }>
// CHECK-LABEL: func private @BSR_explicit(
// CHECK-SAME: tensor<?x?xf64, #[[$BSR_explicit]]>
func.func private @BSR_explicit(%arg0: tensor<?x?xf64, #BSR_explicit>) {
  return
}

// -----

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i            : dense,
    j floordiv 4 : dense,
    j mod 4      : structured[2, 4]
  ),
  crdWidth = 8  // we would even like just 2-bits
}>

// CHECK-DAG: #[[$NV_24:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 floordiv 4 : dense, d1 mod 4 : structured[2, 4]), crdWidth = 8 }>
// CHECK-LABEL: func private @NV_24(
// CHECK-SAME: tensor<?x?xf64, #[[$NV_24]]>
func.func private @NV_24(%arg0: tensor<?x?xf64, #NV_24>) {
  return
}

// -----

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    j            : dense,
    k floordiv 4 : dense,
    k mod 4      : structured[2, 4]
  )
}>

// CHECK-DAG: #[[$NV_24:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : dense, d2 floordiv 4 : dense, d2 mod 4 : structured[2, 4]) }>
// CHECK-LABEL: func private @NV_24(
// CHECK-SAME: tensor<?x?x?xf64, #[[$NV_24]]>
func.func private @NV_24(%arg0: tensor<?x?x?xf64, #NV_24>) {
  return
}

// -----

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 4 : dense,
    j            : dense,
    k mod 4      : structured[2, 4]
  )
}>

// CHECK-DAG: #[[$NV_24:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d2 floordiv 4 : dense, d1 : dense, d2 mod 4 : structured[2, 4]) }>
// CHECK-LABEL: func private @NV_24(
// CHECK-SAME: tensor<?x?x?xf64, #[[$NV_24]]>
func.func private @NV_24(%arg0: tensor<?x?x?xf64, #NV_24>) {
  return
}

// -----

#NOutOfM = #sparse_tensor.encoding<{
  map = ( i, j, k ) ->
  ( i            : dense,
    k floordiv 8 : dense,
    j            : dense,
    k mod 8      : structured[5, 8]
  )
}>

// CHECK-DAG: #[[$NOutOfM:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d2 floordiv 8 : dense, d1 : dense, d2 mod 8 : structured[5, 8]) }>
// CHECK-LABEL: func private @NOutOfM(
// CHECK-SAME: tensor<?x?x?xf64, #[[$NOutOfM]]>
func.func private @NOutOfM(%arg0: tensor<?x?x?xf64, #NOutOfM>) {
  return
}
