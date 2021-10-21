// RUN: mlir-opt %s -arith-expand -split-input-file | FileCheck %s

// Test ceil divide with signed integer
// CHECK-LABEL:       func @ceildivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func @ceildivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = arith.ceildivsi %arg0, %arg1 : i32
  return %res : i32

// CHECK:           [[ONE:%.+]] = arith.constant 1 : i32
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : i32
// CHECK:           [[MINONE:%.+]] = arith.constant -1 : i32
// CHECK:           [[CMP1:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = select [[CMP1]], [[MINONE]], [[ONE]] : i32
// CHECK:           [[TRUE1:%.+]] = arith.addi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = arith.addi [[ONE]], [[TRUE2]] : i32
// CHECK:           [[FALSE1:%.+]] = arith.subi [[ZERO]], [[ARG0]] : i32
// CHECK:           [[FALSE2:%.+]] = arith.divsi [[FALSE1]], [[ARG1]] : i32
// CHECK:           [[FALSE3:%.+]] = arith.subi [[ZERO]], [[FALSE2]] : i32
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MNEG]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MPOS]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE3]] : i32
}

// -----

// Test ceil divide with index type
// CHECK-LABEL:       func @ceildivi_index
// CHECK-SAME:     ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> index {
func @ceildivi_index(%arg0: index, %arg1: index) -> (index) {
  %res = arith.ceildivsi %arg0, %arg1 : index
  return %res : index

// CHECK:           [[ONE:%.+]] = arith.constant 1 : index
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : index
// CHECK:           [[MINONE:%.+]] = arith.constant -1 : index
// CHECK:           [[CMP1:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[X:%.+]] = select [[CMP1]], [[MINONE]], [[ONE]] : index
// CHECK:           [[TRUE1:%.+]] = arith.addi [[X]], [[ARG0]] : index
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : index
// CHECK:           [[TRUE3:%.+]] = arith.addi [[ONE]], [[TRUE2]] : index
// CHECK:           [[FALSE1:%.+]] = arith.subi [[ZERO]], [[ARG0]] : index
// CHECK:           [[FALSE2:%.+]] = arith.divsi [[FALSE1]], [[ARG1]] : index
// CHECK:           [[FALSE3:%.+]] = arith.subi [[ZERO]], [[FALSE2]] : index
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MNEG]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MPOS]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE3]] : index
}

// -----

// Test floor divide with signed integer
// CHECK-LABEL:       func @floordivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func @floordivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = arith.floordivsi %arg0, %arg1 : i32
  return %res : i32
// CHECK:           [[ONE:%.+]] = arith.constant 1 : i32
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : i32
// CHECK:           [[MIN1:%.+]] = arith.constant -1 : i32
// CHECK:           [[CMP1:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = select [[CMP1]], [[ONE]], [[MIN1]] : i32
// CHECK:           [[TRUE1:%.+]] = arith.subi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = arith.subi [[MIN1]], [[TRUE2]] : i32
// CHECK:           [[FALSE:%.+]] = arith.divsi [[ARG0]], [[ARG1]] : i32
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MPOS]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MNEG]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE]] : i32
}

// -----

// Test floor divide with index type
// CHECK-LABEL:       func @floordivi_index
// CHECK-SAME:     ([[ARG0:%.+]]: index, [[ARG1:%.+]]: index) -> index {
func @floordivi_index(%arg0: index, %arg1: index) -> (index) {
  %res = arith.floordivsi %arg0, %arg1 : index
  return %res : index
// CHECK:           [[ONE:%.+]] = arith.constant 1 : index
// CHECK:           [[ZERO:%.+]] = arith.constant 0 : index
// CHECK:           [[MIN1:%.+]] = arith.constant -1 : index
// CHECK:           [[CMP1:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[X:%.+]] = select [[CMP1]], [[ONE]], [[MIN1]] : index
// CHECK:           [[TRUE1:%.+]] = arith.subi [[X]], [[ARG0]] : index
// CHECK:           [[TRUE2:%.+]] = arith.divsi [[TRUE1]], [[ARG1]] : index
// CHECK:           [[TRUE3:%.+]] = arith.subi [[MIN1]], [[TRUE2]] : index
// CHECK:           [[FALSE:%.+]] = arith.divsi [[ARG0]], [[ARG1]] : index
// CHECK:           [[NNEG:%.+]] = arith.cmpi slt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[NPOS:%.+]] = arith.cmpi sgt, [[ARG0]], [[ZERO]] : index
// CHECK:           [[MNEG:%.+]] = arith.cmpi slt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[MPOS:%.+]] = arith.cmpi sgt, [[ARG1]], [[ZERO]] : index
// CHECK:           [[TERM1:%.+]] = arith.andi [[NNEG]], [[MPOS]] : i1
// CHECK:           [[TERM2:%.+]] = arith.andi [[NPOS]], [[MNEG]] : i1
// CHECK:           [[CMP2:%.+]] = arith.ori [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE]] : index
}
