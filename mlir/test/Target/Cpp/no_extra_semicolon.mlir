// RUN: mlir-translate -mlir-to-cpp %s | FileCheck --match-full-lines %s
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck --match-full-lines %s

func.func @no_extra_semicolon(%arg0: i1) {
  emitc.if %arg0 {
    emitc.include "myheader.h"
    emitc.if %arg0 {
    }
    emitc.verbatim "return;"
  }
  return
}
// CHECK: void no_extra_semicolon(bool [[V0:[^ ]*]]) {
// CHECK-NEXT: if ([[V0]]) {
// CHECK-NEXT: #include "myheader.h"
// CHECK-NEXT: if ([[V0]]) {
// CHECK-NEXT: }
// CHECK-NEXT: return;
// CHECK-NEXT: }
// CHECK-NEXT: return;
