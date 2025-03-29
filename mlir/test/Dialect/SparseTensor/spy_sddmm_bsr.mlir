// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification | FileCheck %s

//
// A SDDMM implementation with "spy" function and
// in-place update of the sampling sparse matrix.
//

#BSR = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i floordiv 2 : dense,
    j floordiv 2 : compressed,
    i mod 2 : dense,
    j mod 2 : dense)
}>

#trait_SDDMM = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,  // A
    affine_map<(i,j,k) -> (k,j)>,  // B
    affine_map<(i,j,k) -> (i,j)>   // S (in/out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "S(i,j) += spy[S(i,j)] x SUM_k A(i,k) B(k,j)"
}

//
// CHECK: #[[$BSR:.+]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 2 : dense, d1 floordiv 2 : compressed, d0 mod 2 : dense, d1 mod 2 : dense) }>
// CHECK: #[[$MAP:.+]] = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3) -> (d0 : dense, d1 : compressed, d2 : dense, d3 : dense) }>
//
// CHECK-LABEL:   func.func @SDDMM_block(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf32, #[[$BSR]]>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32, #[[$BSR]]> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.reinterpret_map %[[VAL_0]] : tensor<?x?xf32, #[[$BSR]]> to tensor<?x?x2x2xf32, #[[$MAP]]>
// CHECK-DAG:       %[[VAL_8:.*]] = tensor.dim %[[VAL_1]], %[[VAL_3]] : tensor<?x?xf32>
// CHECK-DAG:       %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_1]] : tensor<?x?xf32> to memref<?x?xf32>
// CHECK-DAG:       %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : tensor<?x?xf32> to memref<?x?xf32>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.lvl %[[VAL_7]], %[[VAL_4]] : tensor<?x?x2x2xf32, #[[$MAP]]>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.positions %[[VAL_7]] {level = 1 : index} : tensor<?x?x2x2xf32, #[[$MAP]]> to memref<?xindex>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_7]] {level = 1 : index} : tensor<?x?x2x2xf32, #[[$MAP]]> to memref<?xindex>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_7]] : tensor<?x?x2x2xf32, #[[$MAP]]> to memref<?xf32>
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_4]] to %[[VAL_11]] step %[[VAL_3]] {
// CHECK:             %[[VAL_16:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : index
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_19:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_3]] {
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_19]]] : memref<?xindex>
// CHECK:               %[[VAL_22:.*]] = arith.muli %[[VAL_19]], %[[VAL_5]] : index
// CHECK:               scf.for %[[VAL_21:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_3]] {
// CHECK:                 %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : index
// CHECK:                 %[[VAL_25:.*]] = arith.muli %[[VAL_23]], %[[VAL_5]] : index
// CHECK:                 scf.for %[[VAL_24:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_3]] {
// CHECK:                   %[[VAL_26:.*]] = arith.addi %[[VAL_24]], %[[VAL_25]] : index
// CHECK:                   %[[VAL_27:.*]] = scf.for %[[VAL_28:.*]] = %[[VAL_4]] to %[[VAL_8]] step %[[VAL_3]] iter_args(%[[VAL_29:.*]] = %[[VAL_6]]) -> (f32) {
// CHECK:                     %[[VAL_30:.*]] = arith.muli %[[VAL_15]], %[[VAL_5]] : index
// CHECK:                     %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_21]] : index
// CHECK:                     %[[VAL_32:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_31]], %[[VAL_28]]] : memref<?x?xf32>
// CHECK:                     %[[VAL_33:.*]] = arith.muli %[[VAL_20]], %[[VAL_5]] : index
// CHECK:                     %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_24]] : index
// CHECK:                     %[[VAL_35:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_28]], %[[VAL_34]]] : memref<?x?xf32>
// CHECK:                     %[[VAL_36:.*]] = arith.mulf %[[VAL_32]], %[[VAL_35]] : f32
// CHECK:                     %[[VAL_37:.*]] = arith.addf %[[VAL_29]], %[[VAL_36]] : f32
// CHECK:                     scf.yield %[[VAL_37]] : f32
// CHECK:                   } {"Emitted from" = "linalg.generic"}
// CHECK:                   memref.store %[[VAL_27]], %[[VAL_14]]{{\[}}%[[VAL_26]]] : memref<?xf32>
// CHECK:                 } {"Emitted from" = "linalg.generic"}
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_38:.*]] = sparse_tensor.load %[[VAL_7]] : tensor<?x?x2x2xf32, #[[$MAP]]>
// CHECK:           %[[VAL_39:.*]] = sparse_tensor.reinterpret_map %[[VAL_38]] : tensor<?x?x2x2xf32, #[[$MAP]]> to tensor<?x?xf32, #[[$BSR]]>
// CHECK:           return %[[VAL_39]] : tensor<?x?xf32, #[[$BSR]]>
// CHECK:         }
module {
  func.func @SDDMM_block(%args: tensor<?x?xf32, #BSR>,
                         %arga: tensor<?x?xf32>,
                         %argb: tensor<?x?xf32>) -> tensor<?x?xf32, #BSR> {
    %result = linalg.generic #trait_SDDMM
      ins(%arga, %argb: tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%args: tensor<?x?xf32, #BSR>) {
        ^bb(%a: f32, %b: f32, %s: f32):
           %f0 = arith.constant 0.0 : f32
           %u = sparse_tensor.unary %s : f32 to f32
             present={
                ^bb0(%p: f32):
                  %mul = arith.mulf %a, %b : f32
                  sparse_tensor.yield %mul : f32
             }
             absent={}
           %r = sparse_tensor.reduce %s, %u, %f0 : f32 {
              ^bb0(%p: f32, %q: f32):
                %add = arith.addf %p, %q : f32
                sparse_tensor.yield %add : f32
            }
           linalg.yield %r : f32
      } -> tensor<?x?xf32, #BSR>
    return %result : tensor<?x?xf32, #BSR>
  }
}
