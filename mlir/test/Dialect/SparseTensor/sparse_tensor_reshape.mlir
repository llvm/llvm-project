// RUN: mlir-opt %s --post-sparsification-rewrite="enable-runtime-library=false enable-convert=false" \
// RUN: --cse --canonicalize  | FileCheck %s

#SparseMatrix = #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>

// CHECK:         func.func @sparse_reshape(
// CHECK-SAME:    %[[S:.*]]:
// CHECK-DAG:     %[[C25:.*]] = arith.constant 25 : index
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[B:.*]] = bufferization.alloc_tensor()
// CHECK:         %[[P0:.*]] = sparse_tensor.positions %[[S]] {level = 0 : index}
// CHECK:         %[[I0:.*]] = sparse_tensor.coordinates %[[S]] {level = 0 : index}
// CHECK:         %[[P1:.*]] = sparse_tensor.positions %[[S]] {level = 1 : index}
// CHECK:         %[[I1:.*]] = sparse_tensor.coordinates %[[S]] {level = 1 : index}
// CHECK:         %[[V:.*]] = sparse_tensor.values %[[S]]
// CHECK:         %[[S0:.*]] = memref.load %[[P0]]{{\[}}%[[C0]]] : memref<?xindex>
// CHECK:         %[[E0:.*]] = memref.load %[[P0]]{{\[}}%[[C1]]] : memref<?xindex>
// CHECK:         %[[RET:.*]] = scf.for %[[I:.*]] = %[[S0]] to %[[E0]] step %[[C1]] iter_args(%[[A0:.*]] = %[[B]])
// CHECK:           %[[SI0:.*]] = memref.load %[[I0]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-DAG:       %[[S1:.*]] = memref.load %[[P1]]{{\[}}%[[I]]] : memref<?xindex>
// CHECK-DAG:       %[[PE1:.*]] = arith.addi %[[I]], %[[C1]] : index
// CHECK:           %[[E1:.*]] = memref.load %[[P1]]{{\[}}%[[PE1]]] : memref<?xindex>
// CHECK:           %[[RET_1:.*]] = scf.for %[[J:.*]] = %[[S1]] to %[[E1]] step %[[C1]] iter_args(%[[A1:.*]] = %[[A0]])
// CHECK:             %[[SI1:.*]] = memref.load %[[I1]]{{\[}}%[[J]]] : memref<?xindex>
// CHECK:             %[[SV:.*]] = memref.load %[[V]]{{\[}}%[[J]]] : memref<?xf64>
// CHECK:             %[[T:.*]] = arith.muli %[[SI0]], %[[C25]] : index
// CHECK:             %[[DI:.*]] = arith.addi %[[T]], %[[SI1]] : index
// CHECK:             %[[D:.*]] = arith.divui %[[DI]], %[[C10]] : index
// CHECK:             %[[R:.*]] = arith.remui %[[DI]], %[[C10]] : index
// CHECK:             %[[R1:.*]] = sparse_tensor.insert %[[SV]] into %[[A1]]{{\[}}%[[D]], %[[R]]]
// CHECK:              scf.yield %[[R1]]
// CHECK:            }
// CHECK:            scf.yield %[[RET_1]]
// CHECK:         }
// CHECK:        %[[NT1:.*]] = sparse_tensor.load %[[RET]] hasInserts
// CHECK:        return %[[NT1]] : tensor<10x10xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed", "compressed" ] }>>
//
func.func @sparse_reshape(%arg0: tensor<4x25xf64, #SparseMatrix>) -> tensor<10x10xf64, #SparseMatrix> {
  %shape = arith.constant dense <[ 10, 10 ]> : tensor<2xi32>
  %0 = tensor.reshape %arg0(%shape) :
    (tensor<4x25xf64, #SparseMatrix>, tensor<2xi32>) -> tensor<10x10xf64, #SparseMatrix>
  return %0 : tensor<10x10xf64, #SparseMatrix>
}
