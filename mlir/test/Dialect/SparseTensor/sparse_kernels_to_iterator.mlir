// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification="sparse-emit-strategy=sparse-iterator" --cse | FileCheck %s --check-prefix="ITER"
// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification="sparse-emit-strategy=sparse-iterator" --cse --sparse-space-collapse --lower-sparse-iteration-to-scf --loop-invariant-code-motion -cse --canonicalize | FileCheck %s



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

// CHECK-LABEL:   func.func @add(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<10xi32, {{.*}}>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<10xi32, {{.*}}>) -> tensor<10xi32> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant dense<0> : tensor<10xi32>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VALUES_0:.*]] = sparse_tensor.values %[[ARG1]] : tensor<10xi32, {{.*}}> to memref<?xi32>
// CHECK:           %[[VALUES_1:.*]] = sparse_tensor.values %[[ARG0]] : tensor<10xi32, {{.*}}> to memref<?xi32>
// CHECK:           %[[TO_BUFFER_0:.*]] = bufferization.to_buffer %[[CONSTANT_2]] : tensor<10xi32> to memref<10xi32>
// CHECK:           linalg.fill ins(%[[CONSTANT_3]] : i32) outs(%[[TO_BUFFER_0]] : memref<10xi32>)
// CHECK:           %[[POSITIONS_0:.*]] = sparse_tensor.positions %[[ARG0]] {level = 0 : index} : tensor<10xi32, {{.*}}> to memref<?xindex>
// CHECK:           %[[COORDINATES_0:.*]] = sparse_tensor.coordinates %[[ARG0]] {level = 0 : index} : tensor<10xi32, {{.*}}> to memref<?xindex>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[POSITIONS_0]]{{\[}}%[[CONSTANT_1]]] : memref<?xindex>
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[POSITIONS_0]]{{\[}}%[[CONSTANT_0]]] : memref<?xindex>
// CHECK:           %[[POSITIONS_1:.*]] = sparse_tensor.positions %[[ARG1]] {level = 0 : index} : tensor<10xi32, {{.*}}> to memref<?xindex>
// CHECK:           %[[COORDINATES_1:.*]] = sparse_tensor.coordinates %[[ARG1]] {level = 0 : index} : tensor<10xi32, {{.*}}> to memref<?xindex>
// CHECK:           %[[LOAD_2:.*]] = memref.load %[[POSITIONS_1]]{{\[}}%[[CONSTANT_1]]] : memref<?xindex>
// CHECK:           %[[LOAD_3:.*]] = memref.load %[[POSITIONS_1]]{{\[}}%[[CONSTANT_0]]] : memref<?xindex>
// CHECK:           %[[WHILE_0:.*]]:2 = scf.while (%[[VAL_0:.*]] = %[[LOAD_0]], %[[VAL_1:.*]] = %[[LOAD_2]]) : (index, index) -> (index, index) {
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi ult, %[[VAL_0]], %[[LOAD_1]] : index
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi ult, %[[VAL_1]], %[[LOAD_3]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:             scf.condition(%[[ANDI_0]]) %[[VAL_0]], %[[VAL_1]] : index, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index):
// CHECK:             %[[LOAD_4:.*]] = memref.load %[[COORDINATES_0]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:             %[[LOAD_5:.*]] = memref.load %[[COORDINATES_1]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi ult, %[[LOAD_5]], %[[LOAD_4]] : index
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[LOAD_5]], %[[LOAD_4]] : index
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi eq, %[[LOAD_4]], %[[SELECT_0]] : index
// CHECK:             %[[CMPI_4:.*]] = arith.cmpi eq, %[[LOAD_5]], %[[SELECT_0]] : index
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_3]], %[[CMPI_4]] : i1
// CHECK:             scf.if %[[ANDI_1]] {
// CHECK:               %[[LOAD_6:.*]] = memref.load %[[VALUES_1]]{{\[}}%[[VAL_2]]] : memref<?xi32>
// CHECK:               %[[LOAD_7:.*]] = memref.load %[[VALUES_0]]{{\[}}%[[VAL_3]]] : memref<?xi32>
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[LOAD_6]], %[[LOAD_7]] : i32
// CHECK:               memref.store %[[ADDI_0]], %[[TO_BUFFER_0]]{{\[}}%[[SELECT_0]]] : memref<10xi32>
// CHECK:             } else {
// CHECK:               scf.if %[[CMPI_3]] {
// CHECK:                 %[[LOAD_8:.*]] = memref.load %[[VALUES_1]]{{\[}}%[[VAL_2]]] : memref<?xi32>
// CHECK:                 memref.store %[[LOAD_8]], %[[TO_BUFFER_0]]{{\[}}%[[SELECT_0]]] : memref<10xi32>
// CHECK:               } else {
// CHECK:                 scf.if %[[CMPI_4]] {
// CHECK:                   %[[LOAD_9:.*]] = memref.load %[[VALUES_0]]{{\[}}%[[VAL_3]]] : memref<?xi32>
// CHECK:                   memref.store %[[LOAD_9]], %[[TO_BUFFER_0]]{{\[}}%[[SELECT_0]]] : memref<10xi32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_0]] : index
// CHECK:             %[[SELECT_1:.*]] = arith.select %[[CMPI_3]], %[[ADDI_1]], %[[VAL_2]] : index
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_0]] : index
// CHECK:             %[[SELECT_2:.*]] = arith.select %[[CMPI_4]], %[[ADDI_2]], %[[VAL_3]] : index
// CHECK:             scf.yield %[[SELECT_1]], %[[SELECT_2]] : index, index
// CHECK:           }
// CHECK:           scf.for %[[VAL_4:.*]] = %[[VAL_5:.*]]#0 to %[[LOAD_1]] step %[[CONSTANT_0]] {
// CHECK:             %[[LOAD_10:.*]] = memref.load %[[COORDINATES_0]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:             %[[LOAD_11:.*]] = memref.load %[[VALUES_1]]{{\[}}%[[VAL_4]]] : memref<?xi32>
// CHECK:             memref.store %[[LOAD_11]], %[[TO_BUFFER_0]]{{\[}}%[[LOAD_10]]] : memref<10xi32>
// CHECK:           }
// CHECK:           scf.for %[[VAL_6:.*]] = %[[VAL_7:.*]]#1 to %[[LOAD_3]] step %[[CONSTANT_0]] {
// CHECK:             %[[LOAD_12:.*]] = memref.load %[[COORDINATES_1]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:             %[[LOAD_13:.*]] = memref.load %[[VALUES_0]]{{\[}}%[[VAL_6]]] : memref<?xi32>
// CHECK:             memref.store %[[LOAD_13]], %[[TO_BUFFER_0]]{{\[}}%[[LOAD_12]]] : memref<10xi32>
// CHECK:           }
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[TO_BUFFER_0]] : memref<10xi32> to tensor<10xi32>
// CHECK:           return %[[TO_TENSOR_0]] : tensor<10xi32>
// CHECK:         }
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
