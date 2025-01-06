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
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10xi32, #sparse{{.*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<10xi32, #sparse{{.*}}>) -> tensor<10xi32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<0> : tensor<10xi32>
// CHECK:           %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_5]] : tensor<10xi32> to memref<10xi32>
// CHECK:           linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_6]] : memref<10xi32>)
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<10xi32, #sparse{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<10xi32, #sparse{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<10xi32, #sparse{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<10xi32, #sparse{{.*}}> to memref<?xindex>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           %[[VAL_15:.*]]:2 = scf.while (%[[VAL_16:.*]] = %[[VAL_9]], %[[VAL_17:.*]] = %[[VAL_13]]) : (index, index) -> (index, index) {
// CHECK:             %[[VAL_18:.*]] = arith.cmpi ult, %[[VAL_16]], %[[VAL_10]] : index
// CHECK:             %[[VAL_19:.*]] = arith.cmpi ult, %[[VAL_17]], %[[VAL_14]] : index
// CHECK:             %[[VAL_20:.*]] = arith.andi %[[VAL_18]], %[[VAL_19]] : i1
// CHECK:             scf.condition(%[[VAL_20]]) %[[VAL_16]], %[[VAL_17]] : index, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_21:.*]]: index, %[[VAL_22:.*]]: index):
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_21]]] : memref<?xindex>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_22]]] : memref<?xindex>
// CHECK:             %[[VAL_25:.*]] = arith.cmpi ult, %[[VAL_24]], %[[VAL_23]] : index
// CHECK:             %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_24]], %[[VAL_23]] : index
// CHECK:             %[[VAL_27:.*]] = arith.cmpi eq, %[[VAL_23]], %[[VAL_26]] : index
// CHECK:             %[[VAL_28:.*]] = arith.cmpi eq, %[[VAL_24]], %[[VAL_26]] : index
// CHECK:             %[[VAL_29:.*]] = arith.andi %[[VAL_27]], %[[VAL_28]] : i1
// CHECK:             scf.if %[[VAL_29]] {
// CHECK:               %[[VAL_30:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10xi32, #sparse{{.*}}> to memref<?xi32>
// CHECK:               %[[VAL_31:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_21]]] : memref<?xi32>
// CHECK:               %[[VAL_32:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<10xi32, #sparse{{.*}}> to memref<?xi32>
// CHECK:               %[[VAL_33:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_22]]] : memref<?xi32>
// CHECK:               %[[VAL_34:.*]] = arith.addi %[[VAL_31]], %[[VAL_33]] : i32
// CHECK:               memref.store %[[VAL_34]], %[[VAL_6]]{{\[}}%[[VAL_26]]] : memref<10xi32>
// CHECK:             } else {
// CHECK:               scf.if %[[VAL_27]] {
// CHECK:                 %[[VAL_35:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10xi32, #sparse{{.*}}> to memref<?xi32>
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_35]]{{\[}}%[[VAL_21]]] : memref<?xi32>
// CHECK:                 memref.store %[[VAL_36]], %[[VAL_6]]{{\[}}%[[VAL_26]]] : memref<10xi32>
// CHECK:               } else {
// CHECK:                 scf.if %[[VAL_28]] {
// CHECK:                   %[[VAL_37:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<10xi32, #sparse{{.*}}> to memref<?xi32>
// CHECK:                   %[[VAL_38:.*]] = memref.load %[[VAL_37]]{{\[}}%[[VAL_22]]] : memref<?xi32>
// CHECK:                   memref.store %[[VAL_38]], %[[VAL_6]]{{\[}}%[[VAL_26]]] : memref<10xi32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             %[[VAL_39:.*]] = arith.addi %[[VAL_21]], %[[VAL_2]] : index
// CHECK:             %[[VAL_40:.*]] = arith.select %[[VAL_27]], %[[VAL_39]], %[[VAL_21]] : index
// CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_22]], %[[VAL_2]] : index
// CHECK:             %[[VAL_42:.*]] = arith.select %[[VAL_28]], %[[VAL_41]], %[[VAL_22]] : index
// CHECK:             scf.yield %[[VAL_40]], %[[VAL_42]] : index, index
// CHECK:           }
// CHECK:           %[[VAL_43:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<10xi32, #sparse{{.*}}> to memref<?xi32>
// CHECK:           scf.for %[[VAL_44:.*]] = %[[VAL_45:.*]]#0 to %[[VAL_10]] step %[[VAL_2]] {
// CHECK:             %[[VAL_46:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_44]]] : memref<?xindex>
// CHECK:             %[[VAL_47:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_44]]] : memref<?xi32>
// CHECK:             memref.store %[[VAL_47]], %[[VAL_6]]{{\[}}%[[VAL_46]]] : memref<10xi32>
// CHECK:           }
// CHECK:           %[[VAL_48:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<10xi32, #sparse{{.*}}> to memref<?xi32>
// CHECK:           scf.for %[[VAL_49:.*]] = %[[VAL_50:.*]]#1 to %[[VAL_14]] step %[[VAL_2]] {
// CHECK:             %[[VAL_51:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_49]]] : memref<?xindex>
// CHECK:             %[[VAL_52:.*]] = memref.load %[[VAL_48]]{{\[}}%[[VAL_49]]] : memref<?xi32>
// CHECK:             memref.store %[[VAL_52]], %[[VAL_6]]{{\[}}%[[VAL_51]]] : memref<10xi32>
// CHECK:           }
// CHECK:           %[[VAL_53:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<10xi32>
// CHECK:           return %[[VAL_53]] : tensor<10xi32>
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
