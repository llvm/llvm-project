// RUN: mlir-opt %s --loop-invariant-code-motion | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : dense,
    j : compressed
  )
}>

// Make sure that pure instructions are hoisted outside the loop.
//
// CHECK: sparse_tensor.values
// CHECK: sparse_tensor.positions
// CHECK: sparse_tensor.coordinate
// CHECK: sparse_tensor.iterate
func.func @sparse_iterate(%sp : tensor<?x?xf64, #CSR>) {
  %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0 : tensor<?x?xf64, #CSR>
                                                         -> !sparse_tensor.iter_space<#CSR, lvls = 0>
  sparse_tensor.iterate %it1 in %l1 at (%crd) : !sparse_tensor.iter_space<#CSR, lvls = 0> {
    %0 = sparse_tensor.values %sp : tensor<?x?xf64, #CSR> to memref<?xf64>
    %1 = sparse_tensor.positions %sp { level = 1 : index } : tensor<?x?xf64, #CSR> to memref<?xindex>
    %2 = sparse_tensor.coordinates  %sp { level = 1 : index } : tensor<?x?xf64, #CSR> to memref<?xindex>
    "test.op"(%0, %1, %2) : (memref<?xf64>, memref<?xindex>, memref<?xindex>) -> ()
  }

  return
}
