// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s

// CHECK-LABEL: func @arrayOperations
// CHECK-SAME:  ([[A0:%.+]]: !smt.bool)
func.func @arrayOperations(%arg0: !smt.bool) {
  // CHECK-NEXT: [[V0:%.+]] = smt.array.broadcast [[A0]] {smt.some_attr} : !smt.array<[!smt.bool -> !smt.bool]>
  %0 = smt.array.broadcast %arg0 {smt.some_attr} : !smt.array<[!smt.bool -> !smt.bool]>
  // CHECK-NEXT: [[V1:%.+]] = smt.array.select [[V0]][[[A0]]] {smt.some_attr} : !smt.array<[!smt.bool -> !smt.bool]>
  %1 = smt.array.select %0[%arg0] {smt.some_attr} : !smt.array<[!smt.bool -> !smt.bool]>
  // CHECK-NEXT: [[V2:%.+]] = smt.array.store [[V0]][[[A0]]], [[A0]] {smt.some_attr} : !smt.array<[!smt.bool -> !smt.bool]>
  %2 = smt.array.store %0[%arg0], %arg0 {smt.some_attr} : !smt.array<[!smt.bool -> !smt.bool]>

  return
}
