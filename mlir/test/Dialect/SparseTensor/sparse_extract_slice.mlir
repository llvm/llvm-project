// RUN: mlir-opt %s --sparse-tensor-codegen --cse |  FileCheck %s

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ]
}>

#CSR_SLICE = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  dimSlices = [ (0, 4, 1), (0, 8, 1) ]
}>

// CHECK-LABEL:   func.func @sparse_slice(
// CHECK-SAME:                            %[[VAL_0:.*0]]: memref<?xindex>,
// CHECK-SAME:                            %[[VAL_1:.*1]]: memref<?xindex>,
// CHECK-SAME:                            %[[VAL_2:.*2]]: memref<?xf64>,
// CHECK-SAME:                            %[[VAL_3:.*3]]: !sparse_tensor.storage_specifier<#sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>)
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.storage_specifier.init with %[[VAL_3]]
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.storage_specifier.set %[[VAL_4]]  dim_offset at 0 with %[[VAL_5]]
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.storage_specifier.set %[[VAL_8]]  lvl_sz at 0 with %[[VAL_6]]
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.storage_specifier.set %[[VAL_9]]  dim_stride at 0 with %[[VAL_7]]
// CHECK:           %[[VAL_11:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.storage_specifier.set %[[VAL_10]]  dim_offset at 1 with %[[VAL_5]]
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.storage_specifier.set %[[VAL_12]]  lvl_sz at 1 with %[[VAL_11]]
// CHECK:           %[[VAL_14:.*]] = sparse_tensor.storage_specifier.set %[[VAL_13]]  dim_stride at 1 with %[[VAL_7]]
// CHECK:           return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_14]]
func.func @sparse_slice(%t1 : tensor<8x8xf64, #CSR>) -> tensor<4x8xf64, #CSR_SLICE> {
  %a1 = tensor.extract_slice %t1[0, 0][4, 8][1, 1] : tensor<8x8xf64, #CSR> to
                                                     tensor<4x8xf64, #CSR_SLICE>
  return %a1 : tensor<4x8xf64, #CSR_SLICE>
}
