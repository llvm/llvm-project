// RUN: mlir-opt %s --sparse-tensor-codegen --canonicalize --cse | FileCheck %s

#CSR = #sparse_tensor.encoding<{ dimLevelType = ["dense", "compressed"]}>
#COO = #sparse_tensor.encoding<{ dimLevelType = ["compressed-nu", "singleton"]}>

// CHECK-LABEL:   func.func @sparse_alloc_copy_CSR(
// CHECK-SAME:      %[[VAL_0:.*0]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_1:.*1]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_2:.*2]]: memref<?xf32>,
// CHECK-SAME:      %[[VAL_3:.*]]: !sparse_tensor.storage_specifier<#{{.*}}>) -> (memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier<#{{.*}}>) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = memref.alloc(%[[VAL_5]]) : memref<?xindex>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_6]] : memref<?xindex> to memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_1]], %[[VAL_4]] : memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = memref.alloc(%[[VAL_7]]) : memref<?xindex>
// CHECK:           memref.copy %[[VAL_1]], %[[VAL_8]] : memref<?xindex> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = memref.dim %[[VAL_2]], %[[VAL_4]] : memref<?xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc(%[[VAL_9]]) : memref<?xf32>
// CHECK:           memref.copy %[[VAL_2]], %[[VAL_10]] : memref<?xf32> to memref<?xf32>
func.func @sparse_alloc_copy_CSR(%arg0: tensor<2x2xf32, #CSR>) -> tensor<2x2xf32, #CSR> {
  %0 = bufferization.alloc_tensor() copy(%arg0) : tensor<2x2xf32, #CSR>
  "test.sink"(%0) : (tensor<2x2xf32, #CSR>) -> ()
}

// CHECK-LABEL:   func.func @sparse_alloc_copy_COO(
// CHECK-SAME:      %[[VAL_0:.*0]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_1:.*1]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_2:.*2]]: memref<?xf32>,
// CHECK-SAME:      %[[VAL_3:.*]]: !sparse_tensor.storage_specifier<#{{.*}}>) -> (memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier<#{{.*}}>) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = memref.alloc(%[[VAL_5]]) : memref<?xindex>
// CHECK:           memref.copy %[[VAL_0]], %[[VAL_6]] : memref<?xindex> to memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_1]], %[[VAL_4]] : memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = memref.alloc(%[[VAL_7]]) : memref<?xindex>
// CHECK:           memref.copy %[[VAL_1]], %[[VAL_8]] : memref<?xindex> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = memref.dim %[[VAL_2]], %[[VAL_4]] : memref<?xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc(%[[VAL_9]]) : memref<?xf32>
// CHECK:           memref.copy %[[VAL_2]], %[[VAL_10]] : memref<?xf32> to memref<?xf32>
func.func @sparse_alloc_copy_COO(%arg0: tensor<2x2xf32, #COO>) -> tensor<2x2xf32, #COO> {
  %0 = bufferization.alloc_tensor() copy(%arg0) : tensor<2x2xf32, #COO>
  "test.sink"(%0) : (tensor<2x2xf32, #COO>) -> ()
}
