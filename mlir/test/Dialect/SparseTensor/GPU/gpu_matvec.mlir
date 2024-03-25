// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --pre-sparsification-rewrite \
// RUN:             --sparse-reinterpret-map \
// RUN:             --sparsification="parallelization-strategy=dense-outer-loop" \
// RUN:             --sparse-gpu-codegen | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

//
// Compute matrix vector y = Ax
//
// CHECK-LABEL: gpu.module @sparse_kernels
// CHECK:       gpu.func @kernel0(
// CHECK-SAME:    %[[VAL_0:.*0]]: index,
// CHECK-SAME:    %[[VAL_1:.*1]]: memref<?xf64>,
// CHECK-SAME:    %[[VAL_2:.*2]]: memref<?xindex>,
// CHECK-SAME:    %[[VAL_3:.*3]]: memref<?xindex>,
// CHECK-SAME:    %[[VAL_4:.*4]]: memref<?xf64>,
// CHECK-SAME:    %[[VAL_5:.*5]]: memref<?xf64>) kernel {
// CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:         %[[VAL_7:.*]] = gpu.block_id  x
// CHECK:         %[[VAL_8:.*]] = gpu.block_dim  x
// CHECK:         %[[VAL_9:.*]] = gpu.thread_id  x
// CHECK:         %[[VAL_10:.*]] = gpu.grid_dim  x
// CHECK:         %[[VAL_11:.*]] = arith.muli %[[VAL_7]], %[[VAL_8]] : index
// CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_9]] : index
// CHECK:         %[[VAL_13:.*]] = arith.muli %[[VAL_8]], %[[VAL_10]] : index
// CHECK:         scf.for %[[VAL_14:.*]] = %[[VAL_12]] to %[[VAL_0]] step %[[VAL_13]] {
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_14]]] : memref<?xf64>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_14]]] : memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_6]] : index
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_6]] iter_args(%[[VAL_21:.*]] = %[[VAL_15]]) -> (f64) {
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_20]]] : memref<?xf64>
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_22]]] : memref<?xf64>
// CHECK:             %[[VAL_25:.*]] = arith.mulf %[[VAL_23]], %[[VAL_24]] : f64
// CHECK:             %[[VAL_26:.*]] = arith.addf %[[VAL_21]], %[[VAL_25]] : f64
// CHECK:             scf.yield %[[VAL_26]] : f64
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           memref.store %[[VAL_27:.*]], %[[VAL_1]]{{\[}}%[[VAL_14]]] : memref<?xf64>
// CHECK:         }
// CHECK:         gpu.return
// CHECK:       }
//
// CHECK-LABEL: func.func @matvec
// CHECK:       gpu.wait async
// CHECK:       gpu.alloc async
// CHECK:       %[[S0:.*]] = gpu.memcpy async
// CHECK:       gpu.wait async
// CHECK:       gpu.alloc async
// CHECK:       %[[S1:.*]] = gpu.memcpy async
// CHECK:       gpu.wait async
// CHECK:       gpu.alloc async
// CHECK:       %[[S2:.*]] = gpu.memcpy async
// CHECK:       gpu.wait async
// CHECK:       gpu.alloc async
// CHECK:       %[[S3:.*]] = gpu.memcpy async
// CHECK:       gpu.wait async
// CHECK:       gpu.alloc async
// CHECK:       %[[S4:.*]] = gpu.memcpy async
// CHECK:       gpu.wait [%[[S0]], %[[S1]], %[[S2]], %[[S3]], %[[S4]]
// CHECK:       %[[T0:.*]] = gpu.launch_func async @sparse_kernels::@kernel0 blocks
// CHECK:       %[[M0:.*]] = gpu.memcpy async [%[[T0]]]
// CHECK:       %[[M1:.*]] = gpu.dealloc async [%[[M0]]]
// CHECK:       %[[M2:.*]] = gpu.wait async
// CHECK:       %[[M3:.*]] = gpu.dealloc async [%[[M2]]]
// CHECK:       %[[M4:.*]] = gpu.wait async
// CHECK:       %[[M5:.*]] = gpu.dealloc async [%[[M4]]]
// CHECK:       %[[M6:.*]] = gpu.wait async
// CHECK:       %[[M7:.*]] = gpu.dealloc async [%[[M6]]]
// CHECK:       %[[M8:.*]] = gpu.wait async
// CHECK:       %[[M9:.*]] = gpu.dealloc async [%[[M8]]]
// CHECK:       gpu.wait [%[[M1]], %[[M3]], %[[M5]], %[[M7]], %[[M9]]
//
func.func @matvec(%A: tensor<?x?xf64, #CSR>, %x: tensor<?xf64>, %y_in: tensor<?xf64>) -> tensor<?xf64> {
  %y_out = linalg.matvec
      ins(%A, %x: tensor<?x?xf64, #CSR>, tensor<?xf64>)
      outs(%y_in: tensor<?xf64>) -> tensor<?xf64>
  return %y_out : tensor<?xf64>
}
