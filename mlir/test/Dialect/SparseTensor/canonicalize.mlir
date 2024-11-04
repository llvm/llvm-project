// RUN: mlir-opt %s -split-input-file -canonicalize="test-convergence" | FileCheck %s

#BCOO = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : loose_compressed(nonunique), d2 : singleton)
}>

// CHECK-DAG: #[[$BCOO:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2) -> (d0 : dense, d1 : loose_compressed(nonunique), d2 : singleton) }>
// CHECK-LABEL: func @sparse_slice_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?xf32, #[[$BCOO]]>
//       CHECK:   %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : tensor<?x?x?xf32, #[[$BCOO]]> to tensor<4x1x?xf32, #[[$BCOO]]>
//       CHECK:   %[[RESULT:.+]] = tensor.cast %[[SLICE]]
//       CHECK:   return %[[RESULT]]
func.func @sparse_slice_canonicalize(%arg0 : tensor<?x?x?xf32, #BCOO>, %arg1 : index,
    %arg2 : index) -> tensor<?x?x?xf32, #BCOO>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.extract_slice %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?x?xf32, #BCOO> to tensor<?x?x?xf32, #BCOO>
  return %0 : tensor<?x?x?xf32, #BCOO>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

// Make sure that the first unused coordinate is optimized.
// CHECK-LABEL: @sparse_iterate_canonicalize
// CHECK:         sparse_tensor.iterate {{.*}} at(_, %{{.*}})
func.func @sparse_iterate_canonicalize(%sp : tensor<?x?xf64, #CSR>) {
  %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0 to 2
      : tensor<?x?xf64, #CSR> -> !sparse_tensor.iter_space<#CSR, lvls = 0 to 2>
  sparse_tensor.iterate %it1 in %l1 at (%coord0, %coord1) : !sparse_tensor.iter_space<#CSR, lvls = 0 to 2> {
    "test.op"(%coord1) : (index) -> ()
  }
  return
}
