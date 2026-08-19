// RUN: mlir-opt %s --sparse-reinterpret-map="loop-ordering-strategy=default" \
// RUN: -sparsification --canonicalize | \
// RUN: FileCheck %s --check-prefixes=DEFAULT,DEFAULT-LOWERED
// RUN: mlir-opt %s --sparse-reinterpret-map="loop-ordering-strategy=dense-outer" \
// RUN: -sparsification --canonicalize | \
// RUN: FileCheck %s --check-prefixes=DENSE,DENSE-LOWERED
// RUN: mlir-opt %s --sparse-reinterpret-map="loop-ordering-strategy=sparse-outer" \
// RUN: -sparsification --canonicalize | \
// RUN: FileCheck %s --check-prefixes=SPARSE,SPARSE-LOWERED

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


// DEFAULT-LOWERED-LABEL: func.func @sparse_loop_ordering_lowered
// DEFAULT-LOWERED-DAG: %[[C0:.*]] = arith.constant 0 : index
// DEFAULT-LOWERED-DAG: %[[C1:.*]] = arith.constant 1 : index
// DEFAULT-LOWERED-DAG: %[[LVL_A:.*]] = sparse_tensor.lvl %arg0, %[[C0]]
// DEFAULT-LOWERED-DAG: %[[POS_A:.*]] = sparse_tensor.positions %arg0 {level = 1 : index}
// DEFAULT-LOWERED-DAG: %[[POS_B:.*]] = sparse_tensor.positions %arg1 {level = 0 : index}
// DEFAULT-LOWERED-DAG: %[[LVL_B:.*]] = sparse_tensor.lvl %arg1, %{{.*}}
// DEFAULT-LOWERED-DAG: %[[DIM_C0:.*]] = tensor.dim %arg2, %[[C0]]
// DEFAULT-LOWERED-DAG: %[[DIM_C1:.*]] = tensor.dim %arg2, %[[C1]]
// DEFAULT-LOWERED-DAG: %[[DIM_C2:.*]] = tensor.dim %arg2, %{{.*}}
// DEFAULT-LOWERED: %[[B_COMPRESSED_START:.*]] = memref.load %[[POS_B]][%[[C0]]]
// DEFAULT-LOWERED: %[[B_COMPRESSED_END:.*]] = memref.load %[[POS_B]][%[[C1]]]
// DEFAULT-LOWERED: scf.for %[[B_COMPRESSED:.*]] = %[[B_COMPRESSED_START]] to %[[B_COMPRESSED_END]] step %[[C1]] {
// DEFAULT-LOWERED:   %[[B_COMPRESSED_PLUS_1:.*]] = arith.addi %[[B_COMPRESSED]], %[[C1]]
// DEFAULT-LOWERED:   scf.for %[[B_SINGLETON:.*]] = %[[B_COMPRESSED]] to %[[B_COMPRESSED_PLUS_1]] step %[[C1]] {
// DEFAULT-LOWERED:     scf.for %[[B_DENSE:.*]] = %[[C0]] to %[[LVL_B]] step %[[C1]] {
// DEFAULT-LOWERED:       scf.for %[[A_DENSE:.*]] = %[[C0]] to %[[LVL_A]] step %[[C1]] {
// DEFAULT-LOWERED:         %[[A_COMPRESSED_START:.*]] = memref.load %[[POS_A]][%[[A_DENSE]]]
// DEFAULT-LOWERED:         %[[A_DENSE_PLUS_1:.*]] = arith.addi %[[A_DENSE]], %[[C1]]
// DEFAULT-LOWERED:         %[[A_COMPRESSED_END:.*]] = memref.load %[[POS_A]][%[[A_DENSE_PLUS_1]]]
// DEFAULT-LOWERED:         scf.for %[[A_COMPRESSED:.*]] = %[[A_COMPRESSED_START]] to %[[A_COMPRESSED_END]] step %[[C1]] {
// DEFAULT-LOWERED:           %[[A_COMPRESSED_PLUS_1:.*]] = arith.addi %[[A_COMPRESSED]], %[[C1]]
// DEFAULT-LOWERED:           scf.for %[[A_SINGLETON:.*]] = %[[A_COMPRESSED]] to %[[A_COMPRESSED_PLUS_1]] step %[[C1]] {
// DEFAULT-LOWERED:             scf.for %[[C_DENSE_0:.*]] = %[[C0]] to %[[DIM_C0]] step %[[C1]] {
// DEFAULT-LOWERED:               scf.for %[[C_DENSE_1:.*]] = %[[C0]] to %[[DIM_C1]] step %[[C1]] {
// DEFAULT-LOWERED:                 scf.for %[[C_DENSE_2:.*]] = %[[C0]] to %[[DIM_C2]] step %[[C1]] {

// DENSE-LOWERED-LABEL: func.func @sparse_loop_ordering_lowered
// DENSE-LOWERED-DAG: %[[C0:.*]] = arith.constant 0 : index
// DENSE-LOWERED-DAG: %[[C1:.*]] = arith.constant 1 : index
// DENSE-LOWERED-DAG: %[[LVL_A:.*]] = sparse_tensor.lvl %arg0, %[[C0]]
// DENSE-LOWERED-DAG: %[[POS_A:.*]] = sparse_tensor.positions %arg0 {level = 1 : index}
// DENSE-LOWERED-DAG: %[[POS_B:.*]] = sparse_tensor.positions %arg1 {level = 0 : index}
// DENSE-LOWERED-DAG: %[[LVL_B:.*]] = sparse_tensor.lvl %arg1, %{{.*}}
// DENSE-LOWERED-DAG: %[[DIM_C0:.*]] = tensor.dim %arg2, %[[C0]]
// DENSE-LOWERED-DAG: %[[DIM_C1:.*]] = tensor.dim %arg2, %[[C1]]
// DENSE-LOWERED-DAG: %[[DIM_C2:.*]] = tensor.dim %arg2, %{{.*}}
// DENSE-LOWERED: scf.for %[[C_DENSE_0:.*]] = %[[C0]] to %[[DIM_C0]] step %[[C1]] {
// DENSE-LOWERED:   scf.for %[[C_DENSE_1:.*]] = %[[C0]] to %[[DIM_C1]] step %[[C1]] {
// DENSE-LOWERED:     scf.for %[[C_DENSE_2:.*]] = %[[C0]] to %[[DIM_C2]] step %[[C1]] {
// DENSE-LOWERED:       scf.for %[[A_DENSE:.*]] = %[[C0]] to %[[LVL_A]] step %[[C1]] iter_args
// DENSE-LOWERED:         %[[A_COMPRESSED_START:.*]] = memref.load %[[POS_A]][%[[A_DENSE]]]
// DENSE-LOWERED:         %[[A_DENSE_PLUS_1:.*]] = arith.addi %[[A_DENSE]], %[[C1]]
// DENSE-LOWERED:         %[[A_COMPRESSED_END:.*]] = memref.load %[[POS_A]][%[[A_DENSE_PLUS_1]]]
// DENSE-LOWERED:         scf.for %[[A_COMPRESSED:.*]] = %[[A_COMPRESSED_START]] to %[[A_COMPRESSED_END]] step %[[C1]] iter_args
// DENSE-LOWERED:           %[[B_COMPRESSED_START:.*]] = memref.load %[[POS_B]][%[[C0]]]
// DENSE-LOWERED:           %[[B_COMPRESSED_END:.*]] = memref.load %[[POS_B]][%[[C1]]]
// DENSE-LOWERED:           scf.for %[[B_COMPRESSED:.*]] = %[[B_COMPRESSED_START]] to %[[B_COMPRESSED_END]] step %[[C1]] iter_args
// DENSE-LOWERED:             %[[A_COMPRESSED_PLUS_1:.*]] = arith.addi %[[A_COMPRESSED]], %[[C1]]
// DENSE-LOWERED:             scf.for %[[A_SINGLETON:.*]] = %[[A_COMPRESSED]] to %[[A_COMPRESSED_PLUS_1]] step %[[C1]] iter_args
// DENSE-LOWERED:               %[[B_COMPRESSED_PLUS_1:.*]] = arith.addi %[[B_COMPRESSED]], %[[C1]]
// DENSE-LOWERED:               scf.for %[[B_SINGLETON:.*]] = %[[B_COMPRESSED]] to %[[B_COMPRESSED_PLUS_1]] step %[[C1]] iter_args
// DENSE-LOWERED:                 scf.for %[[B_DENSE:.*]] = %[[C0]] to %[[LVL_B]] step %[[C1]] iter_args

// SPARSE-LOWERED-LABEL: func.func @sparse_loop_ordering_lowered
// SPARSE-LOWERED-DAG: %[[C0:.*]] = arith.constant 0 : index
// SPARSE-LOWERED-DAG: %[[C1:.*]] = arith.constant 1 : index
// SPARSE-LOWERED-DAG: %[[LVL_A:.*]] = sparse_tensor.lvl %arg0, %[[C0]]
// SPARSE-LOWERED-DAG: %[[POS_A:.*]] = sparse_tensor.positions %arg0 {level = 1 : index}
// SPARSE-LOWERED-DAG: %[[POS_B:.*]] = sparse_tensor.positions %arg1 {level = 0 : index}
// SPARSE-LOWERED-DAG: %[[LVL_B:.*]] = sparse_tensor.lvl %arg1, %{{.*}}
// SPARSE-LOWERED-DAG: %[[DIM_C0:.*]] = tensor.dim %arg2, %[[C0]]
// SPARSE-LOWERED-DAG: %[[DIM_C1:.*]] = tensor.dim %arg2, %[[C1]]
// SPARSE-LOWERED-DAG: %[[DIM_C2:.*]] = tensor.dim %arg2, %{{.*}}
// SPARSE-LOWERED: %[[B_COMPRESSED_START:.*]] = memref.load %[[POS_B]][%[[C0]]]
// SPARSE-LOWERED: %[[B_COMPRESSED_END:.*]] = memref.load %[[POS_B]][%[[C1]]]
// SPARSE-LOWERED: scf.for %[[B_COMPRESSED:.*]] = %[[B_COMPRESSED_START]] to %[[B_COMPRESSED_END]] step %[[C1]] {
// SPARSE-LOWERED:   %[[B_COMPRESSED_PLUS_1:.*]] = arith.addi %[[B_COMPRESSED]], %[[C1]]
// SPARSE-LOWERED:   scf.for %[[B_SINGLETON:.*]] = %[[B_COMPRESSED]] to %[[B_COMPRESSED_PLUS_1]] step %[[C1]] {
// SPARSE-LOWERED:     scf.for %[[C_DENSE_0:.*]] = %[[C0]] to %[[DIM_C0]] step %[[C1]] {
// SPARSE-LOWERED:       scf.for %[[C_DENSE_1:.*]] = %[[C0]] to %[[DIM_C1]] step %[[C1]] {
// SPARSE-LOWERED:         scf.for %[[C_DENSE_2:.*]] = %[[C0]] to %[[DIM_C2]] step %[[C1]] {
// SPARSE-LOWERED:           scf.for %[[A_DENSE:.*]] = %[[C0]] to %[[LVL_A]] step %[[C1]] iter_args
// SPARSE-LOWERED:             %[[A_COMPRESSED_START:.*]] = memref.load %[[POS_A]][%[[A_DENSE]]]
// SPARSE-LOWERED:             %[[A_DENSE_PLUS_1:.*]] = arith.addi %[[A_DENSE]], %[[C1]]
// SPARSE-LOWERED:             %[[A_COMPRESSED_END:.*]] = memref.load %[[POS_A]][%[[A_DENSE_PLUS_1]]]
// SPARSE-LOWERED:             scf.for %[[A_COMPRESSED:.*]] = %[[A_COMPRESSED_START]] to %[[A_COMPRESSED_END]] step %[[C1]] iter_args
// SPARSE-LOWERED:               %[[A_COMPRESSED_PLUS_1:.*]] = arith.addi %[[A_COMPRESSED]], %[[C1]]
// SPARSE-LOWERED:               scf.for %[[A_SINGLETON:.*]] = %[[A_COMPRESSED]] to %[[A_COMPRESSED_PLUS_1]] step %[[C1]] iter_args
// SPARSE-LOWERED:                 scf.for %[[B_DENSE:.*]] = %[[C0]] to %[[LVL_B]] step %[[C1]] iter_args

func.func @sparse_loop_ordering_lowered(%A: tensor<?x?x?xf32, #X>,
                                        %B: tensor<?x?x?xf32, #Y>,
                                        %C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %result = linalg.generic #trait
  ins(%A, %B: tensor<?x?x?xf32, #X>, tensor<?x?x?xf32, #Y>)
     outs(%C: tensor<?x?x?xf32>) {
    ^bb(%a: f32, %b: f32, %c: f32):
      %ab = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %ab : f32
      linalg.yield %sum : f32
  } -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}
