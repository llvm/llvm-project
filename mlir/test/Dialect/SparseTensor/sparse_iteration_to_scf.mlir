// RUN: mlir-opt %s --lower-sparse-iteration-to-scf | FileCheck %s

#COO = #sparse_tensor.encoding<{
  map = (i, j) -> (
    i : compressed(nonunique),
    j : singleton(soa)
  )
}>

// CHECK-LABEL:   func.func @sparse_1D_space(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf32, #sparse{{[0-9]*}}>) -> !sparse_tensor.iter_space<#sparse{{[0-9]*}}, lvls = 0> {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[LVL_SIZE:.*]] = sparse_tensor.lvl %[[VAL_0]], %[[C0]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// CHECK:           %[[POS_MEM:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK:           %[[CRD_MEM:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK:           %[[POS_LO:.*]] = memref.load %[[POS_MEM]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:           %[[POS_HI:.*]] = memref.load %[[POS_MEM]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:           %[[ITER_SPACE:.*]] = builtin.unrealized_conversion_cast %[[POS_MEM]], %[[CRD_MEM]], %[[LVL_SIZE]], %[[POS_LO]], %[[POS_HI]]
func.func @sparse_1D_space(%sp : tensor<?x?xf32, #COO>) -> !sparse_tensor.iter_space<#COO, lvls = 0> {
  %l1 = sparse_tensor.extract_iteration_space %sp lvls = 0 : tensor<?x?xf32, #COO> -> !sparse_tensor.iter_space<#COO, lvls = 0>
  return %l1 : !sparse_tensor.iter_space<#COO, lvls = 0>
}
