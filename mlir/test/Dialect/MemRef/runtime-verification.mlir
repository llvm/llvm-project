// RUN: mlir-opt %s -generate-runtime-verification -cse -split-input-file | FileCheck %s

// CHECK-LABEL: func @expand_shape(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>
//  CHECK-SAME:     %[[sz0:.*]]: index 
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:   %[[dim:.*]] = memref.dim %[[m]], %[[c0]]
//       CHECK:   %[[mod:.*]] = arith.remsi %[[dim]], %[[c5]]
//       CHECK:   %[[cmpi:.*]] = arith.cmpi eq, %[[mod]], %[[c0]]
//       CHECK:   cf.assert %[[cmpi]], "ERROR: Runtime op verification failed
func.func @expand_shape(%m: memref<?xf32>, %sz0: index) -> memref<?x5xf32> {
  %0 = memref.expand_shape %m [[0, 1]] output_shape [%sz0, 5] : memref<?xf32> into memref<?x5xf32>
  return %0 : memref<?x5xf32>
}

// -----

// CHECK-LABEL:   func.func @subview(
// CHECK-SAME:      %[[ARG0:.*]]: memref<1xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: index) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_0:.*]], %[[VAL_1:.*]], %[[VAL_2:.*]], %[[EXTRACT_STRIDED_METADATA_0:.*]] = memref.extract_strided_metadata %[[ARG0]] : memref<1xf32> -> memref<f32>, index, index, index
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[CONSTANT_1]], %[[CONSTANT_0]] : index
// CHECK:           %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (i1) {
// CHECK:             %[[CMPI_1:.*]] = arith.cmpi sge, %[[ARG1]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_2:.*]] = arith.cmpi sle, %[[ARG1]], %[[VAL_2]] : index
// CHECK:             %[[ANDI_0:.*]] = arith.andi %[[CMPI_1]], %[[CMPI_2]] : i1
// CHECK:             scf.yield %[[ANDI_0]] : i1
// CHECK:           } else {
// CHECK:             %[[CMPI_3:.*]] = arith.cmpi sge, %[[ARG1]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_4:.*]] = arith.cmpi slt, %[[ARG1]], %[[VAL_2]] : index
// CHECK:             %[[ANDI_1:.*]] = arith.andi %[[CMPI_3]], %[[CMPI_4]] : i1
// CHECK:             scf.yield %[[ANDI_1]] : i1
// CHECK:           }
// CHECK:           cf.assert %[[IF_0]], "ERROR: Runtime op verification failed
// CHECK:           %[[CMPI_5:.*]] = arith.cmpi sgt, %[[CONSTANT_1]], %[[CONSTANT_0]] : index
// CHECK:           %[[IF_1:.*]] = scf.if %[[CMPI_5]] -> (i1) {
// CHECK:             %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_1]], %[[CONSTANT_1]] : index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[ARG1]], %[[MULI_0]] : index
// CHECK:             %[[CMPI_6:.*]] = arith.cmpi sge, %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[CMPI_7:.*]] = arith.cmpi slt, %[[ADDI_0]], %[[VAL_2]] : index
// CHECK:             %[[ANDI_2:.*]] = arith.andi %[[CMPI_6]], %[[CMPI_7]] : i1
// CHECK:             scf.yield %[[ANDI_2]] : i1
// CHECK:           } else {
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant true
// CHECK:             scf.yield %[[CONSTANT_2]] : i1
// CHECK:           }
// CHECK:           cf.assert %[[IF_1]], "ERROR: Runtime op verification failed
func.func @subview(%memref: memref<1xf32>, %offset: index) {
  memref.subview %memref[%offset] [1] [1] : 
      memref<1xf32> to 
      memref<1xf32, strided<[1], offset: ?>>
  return
}
