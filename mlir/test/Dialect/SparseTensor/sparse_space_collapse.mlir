// RUN: mlir-opt %s --sparse-space-collapse | FileCheck %s

#COO = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : compressed(nonunique),
    j : singleton(soa)
  )
}>

// CHECK-LABEL:   func.func @sparse_sparse_collapse(
// CHECK-SAME:        %[[VAL_0:.*]]: tensor<4x8xf32, #sparse>) -> index {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = sparse_tensor.extract_iteration_space %[[VAL_0]] lvls = 0 to 2
// CHECK:           %[[VAL_3:.*]] = sparse_tensor.iterate %[[VAL_4:.*]] in %[[VAL_2]]
// CHECK-SAME:                                                          at(%[[VAL_5:.*]], _)
// CHECK-SAME:                                                    iter_args(%[[VAL_6:.*]] = %[[VAL_1]])
// CHECK-SAME:                    : !sparse_tensor.iter_space<#sparse, lvls = 0 to 2> -> (index) {
// CHECK:             %[[VAL_7:.*]] = "test.op"(%[[VAL_6]]) : (index) -> index
// CHECK:             sparse_tensor.yield %[[VAL_7]] : index
// CHECK:           }
// CHECK:           return %[[VAL_3]] : index
// CHECK:         }
func.func @sparse_sparse_collapse(%sp : tensor<4x8xf32, #COO>) -> index {
  %i = arith.constant 0 : index
  %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0
      : tensor<4x8xf32, #COO>
     -> !sparse_tensor.iter_space<#COO, lvls = 0>
    %r1 = sparse_tensor.iterate %it1 in %l1 at(%coord) iter_args(%outer = %i): !sparse_tensor.iter_space<#COO, lvls = 0 to 1> -> index {
    %l2 = sparse_tensor.extract_iteration_space %sp at %it1 lvls = 1
        : tensor<4x8xf32, #COO>, !sparse_tensor.iterator<#COO, lvls = 0 to 1>
       -> !sparse_tensor.iter_space<#COO, lvls = 1>
    %r2 = sparse_tensor.iterate %it2 in %l2 iter_args(%inner = %outer): !sparse_tensor.iter_space<#COO, lvls = 1 to 2> -> index {
      %k = "test.op"(%inner) : (index) -> index
      sparse_tensor.yield %k : index
    }
    sparse_tensor.yield %r2 : index
  }
  return %r1 : index
}
