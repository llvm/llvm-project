// RUN: mlir-opt %s --lower-sparse-ops-to-foreach --canonicalize | FileCheck %s
// RUN: mlir-opt %s --lower-sparse-ops-to-foreach='enable-runtime-library=false' --canonicalize | FileCheck %s

#Loose = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : loose_compressed)
}>
#LooseDense = #sparse_tensor.encoding<{
  map = (i, j, k) -> (i : dense, j : loose_compressed, k : dense)
}>
#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

// CHECK-LABEL: func.func @loose(
// CHECK-SAME: %[[T:.*]]: tensor
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[COUNT:.*]] = sparse_tensor.foreach in %[[T]] init(%[[ZERO]])
// CHECK: ^bb0({{.*}}, %[[ACC:.*]]: index):
// CHECK: %[[NEXT:.*]] = arith.addi %[[ACC]], %[[ONE]] : index
// CHECK: sparse_tensor.yield %[[NEXT]] : index
// CHECK: return %[[COUNT]] : index
func.func @loose(%t: tensor<?x?xf64, #Loose>) -> index {
  %n = sparse_tensor.number_of_entries %t : tensor<?x?xf64, #Loose>
  return %n : index
}

// Check all levels, not only the innermost one.
// CHECK-LABEL: func.func @loose_dense(
// CHECK: sparse_tensor.foreach
// CHECK-NOT: sparse_tensor.number_of_entries
// CHECK: return
func.func @loose_dense(%t: tensor<?x?x2xf64, #LooseDense>) -> index {
  %n = sparse_tensor.number_of_entries %t : tensor<?x?x2xf64, #LooseDense>
  return %n : index
}

// Formats with contiguous storage retain their constant-time query.
// CHECK-LABEL: func.func @compressed(
// CHECK-NOT: sparse_tensor.foreach
// CHECK: %[[N:.*]] = sparse_tensor.number_of_entries
// CHECK: return %[[N]] : index
func.func @compressed(%t: tensor<?x?xf64, #CSR>) -> index {
  %n = sparse_tensor.number_of_entries %t : tensor<?x?xf64, #CSR>
  return %n : index
}
