// RUN: mlir-opt %s --sparse-space-collapse | FileCheck %s

#COO = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : compressed(nonunique),
    j : singleton(soa)
  )
}>

// CHECK-LABEL:   func.func @sparse_sparse_collapse(
// CHECK-SAME:         %[[VAL_0:.*]]: tensor<4x8xf32, #sparse>,
// CHECK-SAME:         %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_3:.*]] = sparse_tensor.extract_iteration_space %[[VAL_0]] lvls = 0 to 2 : tensor<4x8xf32, #sparse>
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.iterate %[[VAL_5:.*]] in %[[VAL_3]] at(%[[VAL_6:.*]], _) iter_args(%[[VAL_7:.*]] = %[[VAL_1]])
// CHECK:             %[[VAL_8:.*]] = "test.op"(%[[VAL_7]]) : (index) -> index
// CHECK:             sparse_tensor.yield %[[VAL_8]] : index
// CHECK:           }
// CHECK:           "test.sink"(%[[VAL_4]]) : (index) -> ()
// CHECK:           return
// CHECK:         }
func.func @sparse_sparse_collapse(%sp : tensor<4x8xf32, #COO>, %i : index) {
  %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0
      : tensor<4x8xf32, #COO>
     -> !sparse_tensor.iter_space<#COO, lvls = 0>
  %r1 = sparse_tensor.iterate %it1 in %l1 at(%crd0) iter_args(%outer = %i): !sparse_tensor.iter_space<#COO, lvls = 0 to 1> -> index {
    %l2 = sparse_tensor.extract_iteration_space %sp at %it1 lvls = 1
        : tensor<4x8xf32, #COO>, !sparse_tensor.iterator<#COO, lvls = 0 to 1>
       -> !sparse_tensor.iter_space<#COO, lvls = 1>
    %r2 = sparse_tensor.iterate %it2 in %l2 iter_args(%inner = %outer): !sparse_tensor.iter_space<#COO, lvls = 1 to 2> -> index {
      %k ="test.op"(%inner) : (index) -> index
      sparse_tensor.yield %k : index
    }
    sparse_tensor.yield %r2 : index
  }
  "test.sink"(%r1) : (index) -> ()
  return
}
