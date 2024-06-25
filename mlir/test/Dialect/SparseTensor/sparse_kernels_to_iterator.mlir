// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification="sparse-emit-strategy=sparse-iterator" --sparse-space-collapse --lower-sparse-iteration-to-scf | FileCheck %s


#COO = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (
    d0 : compressed(nonunique),
    d1 : singleton(nonunique, soa),
    d2 : singleton(nonunique, soa),
    d3 : singleton(soa)
  ),
  explicitVal = 1 : i32
}>

// CHECK-LABEL:   func.func @sqsum(
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[POS_BUF:.*]] = sparse_tensor.positions %{{.*}} {level = 0 : index} : tensor<?x?x?x?xi32, #sparse> to memref<?xindex>
// CHECK:           %[[POS_LO:.*]] = memref.load %[[POS_BUF]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:           %[[POS_HI:.*]] = memref.load %[[POS_BUF]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:           %[[SQ_SUM:.*]] = scf.for %[[POS:.*]] = %[[POS_LO]] to %[[POS_HI]] step %[[C1]] {{.*}} {
// CHECK:             %[[SUM:.*]] = arith.addi
// CHECK:             scf.yield %[[SUM]] : i32
// CHECK:           }
// CHECK:           memref.store
// CHECK:           %[[RET:.*]] = bufferization.to_tensor
// CHECK:           return %[[RET]] : tensor<i32>
// CHECK:         }
func.func @sqsum(%arg0: tensor<?x?x?x?xi32, #COO>) -> tensor<i32> {
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> ()>
    ],
    iterator_types = ["reduction", "reduction", "reduction", "reduction"]
  } ins(%arg0 : tensor<?x?x?x?xi32, #COO>) outs(%cst : tensor<i32>) {
  ^bb0(%in: i32, %out: i32):
    %1 = arith.muli %in, %in : i32
    %2 = arith.addi %out, %1 : i32
    linalg.yield %2 : i32
  } -> tensor<i32>
  return %0 : tensor<i32>
}
