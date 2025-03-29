// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --pre-sparsification-rewrite \
// RUN:             --sparse-reinterpret-map \
// RUN:             --sparsification="parallelization-strategy=dense-outer-loop" \
// RUN:             --sparse-gpu-codegen | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

//
// Compute matrix matrix C = AB
//
// CHECK-LABEL: gpu.module @sparse_kernels
// CHECK-LABEL: gpu.func @kernel0(
// CHECK-SAME:    %[[VAL_0:.*0]]: index,
// CHECK-SAME:    %[[VAL_1:.*1]]: index,
// CHECK-SAME:    %[[VAL_2:.*2]]: memref<?xindex>,
// CHECK-SAME:    %[[VAL_3:.*3]]: memref<?xindex>,
// CHECK-SAME:    %[[VAL_4:.*4]]: memref<?xf64>,
// CHECK-SAME:    %[[VAL_5:.*5]]: memref<?x?xf64>,
// CHECK-SAME:    %[[VAL_6:.*6]]: memref<?x?xf64>) kernel {
// CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:         %[[VAL_9:.*]] = gpu.block_id  x
// CHECK:         %[[VAL_10:.*]] = gpu.block_dim  x
// CHECK:         %[[VAL_11:.*]] = gpu.thread_id  x
// CHECK:         %[[VAL_12:.*]] = gpu.grid_dim  x
// CHECK:         %[[VAL_13:.*]] = arith.muli %[[VAL_9]], %[[VAL_10]] : index
// CHECK:         %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_11]] : index
// CHECK:         %[[VAL_15:.*]] = arith.muli %[[VAL_10]], %[[VAL_12]] : index
// CHECK:         scf.for %[[VAL_16:.*]] = %[[VAL_14]] to %[[VAL_1]] step %[[VAL_15]] {
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_7]] : index
// CHECK:           %[[VAL_19:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_7]] {
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_20]]] : memref<?xf64>
// CHECK:             scf.for %[[VAL_23:.*]] = %[[VAL_8]] to %[[VAL_0]] step %[[VAL_7]] {
// CHECK:               %[[VAL_24:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_16]], %[[VAL_23]]] : memref<?x?xf64>
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_21]], %[[VAL_23]]] : memref<?x?xf64>
// CHECK:               %[[VAL_26:.*]] = arith.mulf %[[VAL_22]], %[[VAL_25]] : f64
// CHECK:               %[[VAL_27:.*]] = arith.addf %[[VAL_24]], %[[VAL_26]] : f64
// CHECK:               memref.store %[[VAL_27]], %[[VAL_5]]{{\[}}%[[VAL_16]], %[[VAL_23]]] : memref<?x?xf64>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:         }
// CHECK:         gpu.return
// CHECK:       }
//
//
// CHECK-LABEL: func.func @matmul
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
func.func @matmul(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xf64>, %C_in: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %C_out = linalg.matmul
      ins(%A, %B: tensor<?x?xf64, #CSR>, tensor<?x?xf64>)
      outs(%C_in: tensor<?x?xf64>) -> tensor<?x?xf64>
  return %C_out : tensor<?x?xf64>
}
