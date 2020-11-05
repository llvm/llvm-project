// RUN: mlir-opt -std-expand-divs %s -split-input-file | FileCheck %s

// Test floor divide with signed integer
// CHECK-LABEL:       func @floordivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func @floordivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = floordivi_signed %arg0, %arg1 : i32
  return %res : i32
// CHECK:           [[ONE:%.+]] = constant 1 : i32
// CHECK:           [[ZERO:%.+]] = constant 0 : i32
// CHECK:           [[MIN1:%.+]] = constant -1 : i32
// CHECK:           [[CMP1:%.+]] = cmpi "slt", [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = select [[CMP1]], [[ONE]], [[MIN1]] : i32
// CHECK:           [[TRUE1:%.+]] = subi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = divi_signed [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = subi [[MIN1]], [[TRUE2]] : i32
// CHECK:           [[FALSE:%.+]] = divi_signed [[ARG0]], [[ARG1]] : i32
// CHECK:           [[NNEG:%.+]] = cmpi "slt", [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = cmpi "sgt", [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = cmpi "slt", [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = cmpi "sgt", [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = and [[NNEG]], [[MPOS]] : i1
// CHECK:           [[TERM2:%.+]] = and [[NPOS]], [[MNEG]] : i1
// CHECK:           [[CMP2:%.+]] = or [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE]] : i32
}

// -----

// Test ceil divide with signed integer
// CHECK-LABEL:       func @ceildivi
// CHECK-SAME:     ([[ARG0:%.+]]: i32, [[ARG1:%.+]]: i32) -> i32 {
func @ceildivi(%arg0: i32, %arg1: i32) -> (i32) {
  %res = ceildivi_signed %arg0, %arg1 : i32
  return %res : i32

// CHECK:           [[ONE:%.+]] = constant 1 : i32
// CHECK:           [[ZERO:%.+]] = constant 0 : i32
// CHECK:           [[MINONE:%.+]] = constant -1 : i32
// CHECK:           [[CMP1:%.+]] = cmpi "sgt", [[ARG1]], [[ZERO]] : i32
// CHECK:           [[X:%.+]] = select [[CMP1]], [[MINONE]], [[ONE]] : i32
// CHECK:           [[TRUE1:%.+]] = addi [[X]], [[ARG0]] : i32
// CHECK:           [[TRUE2:%.+]] = divi_signed [[TRUE1]], [[ARG1]] : i32
// CHECK:           [[TRUE3:%.+]] = addi [[ONE]], [[TRUE2]] : i32
// CHECK:           [[FALSE1:%.+]] = subi [[ZERO]], [[ARG0]] : i32
// CHECK:           [[FALSE2:%.+]] = divi_signed [[FALSE1]], [[ARG1]] : i32
// CHECK:           [[FALSE3:%.+]] = subi [[ZERO]], [[FALSE2]] : i32
// CHECK:           [[NNEG:%.+]] = cmpi "slt", [[ARG0]], [[ZERO]] : i32
// CHECK:           [[NPOS:%.+]] = cmpi "sgt", [[ARG0]], [[ZERO]] : i32
// CHECK:           [[MNEG:%.+]] = cmpi "slt", [[ARG1]], [[ZERO]] : i32
// CHECK:           [[MPOS:%.+]] = cmpi "sgt", [[ARG1]], [[ZERO]] : i32
// CHECK:           [[TERM1:%.+]] = and [[NNEG]], [[MNEG]] : i1
// CHECK:           [[TERM2:%.+]] = and [[NPOS]], [[MPOS]] : i1
// CHECK:           [[CMP2:%.+]] = or [[TERM1]], [[TERM2]] : i1
// CHECK:           [[RES:%.+]] = select [[CMP2]], [[TRUE3]], [[FALSE3]] : i32
}
