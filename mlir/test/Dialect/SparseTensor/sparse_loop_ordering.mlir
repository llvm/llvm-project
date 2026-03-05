// RUN: mlir-opt %s --sparse-reinterpret-map="loop-ordering-strategy=default" \
// RUN: -sparsification --canonicalize | \
// RUN: FileCheck %s --check-prefix=DEFAULT
// RUN: mlir-opt %s --sparse-reinterpret-map="loop-ordering-strategy=dense-outer" \
// RUN: -sparsification --canonicalize | \
// RUN: FileCheck %s --check-prefix=DENSE
// RUN: mlir-opt %s --sparse-reinterpret-map="loop-ordering-strategy=sparse-outer" \
// RUN: -sparsification --canonicalize | \
// RUN: FileCheck %s --check-prefix=SPARSE

#X = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (
    d0 : dense,
    d1 : compressed,
    d2 : singleton)
}>

#Y = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (
    d0 : compressed,
    d1 : singleton,
    d2 : dense)
}>

#trait = {
  indexing_maps = [
    affine_map<(i,j,k,l,m,n,o,p,q) -> (l,m,n)>,
    affine_map<(i,j,k,l,m,n,o,p,q) -> (o,p,q)>,
    affine_map<(i,j,k,l,m,n,o,p,q) -> (i,j,k)>
  ],
  iterator_types = ["parallel", "parallel", "parallel", 
                    "parallel", "parallel", "parallel",
                    "parallel", "parallel", "parallel"]
}

// DEFAULT: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d3, d4, d5)>
// DEFAULT: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2)>
// DEFAULT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d6, d7, d8)>
// DEFAULT-LABEL: func.func @sparse_loop_ordering
// DEFAULT: linalg.generic
// DEFAULT-SAME: sorted = true

// DENSE: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d3, d6)>
// DENSE: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d4, d7, d8)>
// DENSE: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5)>
// DENSE-LABEL: func.func @sparse_loop_ordering
// DENSE: linalg.generic
// DENSE-SAME: sorted = true

// SPARSE: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d5, d6, d7)>
// SPARSE: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d8)>
// SPARSE: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d3, d4)>
// SPARSE-LABEL: func.func @sparse_loop_ordering
// SPARSE: linalg.generic
// SPARSE-SAME: sorted = true

func.func @sparse_loop_ordering(%A: tensor<?x?x?xf32, #X>,
                                %B: tensor<?x?x?xf32, #Y>,
                                %C: tensor<?x?x?xf32, #X>) -> tensor<?x?x?xf32, #X> {
  %result = linalg.generic #trait
  ins(%A, %B: tensor<?x?x?xf32, #X>, tensor<?x?x?xf32, #Y>)
     outs(%C: tensor<?x?x?xf32, #X>) {
    ^bb(%a: f32, %b: f32, %c: f32):
      %ab = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %ab : f32
      linalg.yield %sum : f32
  } -> tensor<?x?x?xf32, #X>
  return %result : tensor<?x?x?xf32, #X>
}
