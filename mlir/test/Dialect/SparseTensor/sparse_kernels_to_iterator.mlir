// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification="sparse-emit-strategy=sparse-iterator" --cse | FileCheck %s --check-prefix="ITER"

// TODO: temporarilly disabled since there is no lowering rules from `coiterate` to `scf`.
// R_U_N: mlir-opt %s --sparse-reinterpret-map -sparsification="sparse-emit-strategy=sparse-iterator" --cse --sparse-space-collapse --lower-sparse-iteration-to-scf --loop-invariant-code-motion | FileCheck %s



#COO = #sparse_tensor.encoding<{
  map = (d0, d1, d2, d3) -> (
    d0 : compressed(nonunique),
    d1 : singleton(nonunique, soa),
    d2 : singleton(nonunique, soa),
    d3 : singleton(soa)
  )
}>

#VEC = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>


// CHECK-LABEL:   func.func @sqsum(
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[POS_BUF:.*]] = sparse_tensor.positions %{{.*}} {level = 0 : index} : tensor<?x?x?x?xi32, #sparse{{.*}}> to memref<?xindex>
// CHECK:           %[[POS_LO:.*]] = memref.load %[[POS_BUF]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:           %[[POS_HI:.*]] = memref.load %[[POS_BUF]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:           %[[VAL_BUF:.*]] = sparse_tensor.values %{{.*}} : tensor<?x?x?x?xi32, #sparse{{.*}}> to memref<?xi32>
// CHECK:           %[[SQ_SUM:.*]] = scf.for %[[POS:.*]] = %[[POS_LO]] to %[[POS_HI]] step %[[C1]] {{.*}} {
// CHECK:             %[[VAL:.*]] = memref.load %[[VAL_BUF]]{{\[}}%[[POS]]] : memref<?xi32>
// CHECK:             %[[MUL:.*]] = arith.muli %[[VAL]], %[[VAL]] : i32
// CHECK:             %[[SUM:.*]] = arith.addi
// CHECK:             scf.yield %[[SUM]] : i32
// CHECK:           }
// CHECK:           memref.store
// CHECK:           %[[RET:.*]] = bufferization.to_tensor
// CHECK:           return %[[RET]] : tensor<i32>
// CHECK:         }

// ITER-LABEL:   func.func @sqsum(
// ITER:           sparse_tensor.iterate
// ITER:             sparse_tensor.iterate
// ITER:               sparse_tensor.iterate
// ITER:         }
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


// ITER-LABEL:   func.func @add(
// ITER:           sparse_tensor.coiterate
// ITER:           case %[[IT_1:.*]], %[[IT_2:.*]] {
// ITER:             %[[LHS:.*]] = sparse_tensor.extract_value %{{.*}} at %[[IT_1]]
// ITER:             %[[RHS:.*]] = sparse_tensor.extract_value %{{.*}} at %[[IT_2]]
// ITER:             %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i32
// ITER:             memref.store %[[SUM]]
// ITER:           }
// ITER:           case %[[IT_1:.*]], _ {
// ITER:             %[[LHS:.*]] = sparse_tensor.extract_value %{{.*}} at %[[IT_1]]
// ITER:             memref.store %[[LHS]]
// ITER:           }
// ITER:           case _, %[[IT_2:.*]] {
// ITER:             %[[RHS:.*]] = sparse_tensor.extract_value %{{.*}} at %[[IT_2]]
// ITER:             memref.store %[[RHS]]
// ITER:           }
// ITER:           bufferization.to_tensor
// ITER:           return
// ITER:         }
func.func @add(%arg0: tensor<10xi32, #VEC>, %arg1: tensor<10xi32, #VEC>) -> tensor<10xi32> {
  %cst = arith.constant dense<0> : tensor<10xi32>
  %0 = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%arg0, %arg1 : tensor<10xi32, #VEC>, tensor<10xi32, #VEC>)
  outs(%cst : tensor<10xi32>) {
    ^bb0(%in1: i32, %in2: i32, %out: i32):
      %2 = arith.addi %in1, %in2 : i32
      linalg.yield %2 : i32
  } -> tensor<10xi32>
  return %0 : tensor<10xi32>
}
