// RUN: mlir-opt %s --lower-sparse-iteration-to-scf | FileCheck %s

#COO = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : compressed(nonunique),
    j : singleton(soa)
  )
}>

func.func @sparse_sparse_collapse(%sp : tensor<4x8xf32, #COO>) -> index {
  %i = arith.constant 0 : index
  %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0
      : tensor<4x8xf32, #COO>
     -> !sparse_tensor.iter_space<#COO, lvls = 0>
    %r1 = sparse_tensor.iterate %it1 in %l1 iter_args(%outer = %i): !sparse_tensor.iter_space<#COO, lvls = 0 to 1> -> index {
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
