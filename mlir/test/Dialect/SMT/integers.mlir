// RUN: mlir-opt %s --verify-roundtrip | FileCheck %s

// CHECK-LABEL: func @integer_operations
func.func @integer_operations() {
  // CHECK-NEXT: [[V0:%.+]] = smt.int.constant -123 {smt.some_attr}
  %0 = smt.int.constant -123 {smt.some_attr}
  // CHECK-NEXT: %c184467440737095516152 = smt.int.constant 184467440737095516152 {smt.some_attr}
  %1 = smt.int.constant 184467440737095516152 {smt.some_attr}


  // CHECK-NEXT: smt.int.add [[V0]], [[V0]], [[V0]] {smt.some_attr}
  %2 = smt.int.add %0, %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.mul [[V0]], [[V0]], [[V0]] {smt.some_attr}
  %3 = smt.int.mul %0, %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.sub [[V0]], [[V0]] {smt.some_attr}
  %4 = smt.int.sub %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.div [[V0]], [[V0]] {smt.some_attr}
  %5 = smt.int.div %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.mod [[V0]], [[V0]] {smt.some_attr}
  %6 = smt.int.mod %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.abs [[V0]] {smt.some_attr}
  %7 = smt.int.abs %0 {smt.some_attr}

  // CHECK-NEXT: smt.int.cmp le [[V0]], [[V0]] {smt.some_attr}
  %9 = smt.int.cmp le %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.cmp lt [[V0]], [[V0]] {smt.some_attr}
  %10 = smt.int.cmp lt %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.cmp ge [[V0]], [[V0]] {smt.some_attr}
  %11 = smt.int.cmp ge %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int.cmp gt [[V0]], [[V0]] {smt.some_attr}
  %12 = smt.int.cmp gt %0, %0 {smt.some_attr}
  // CHECK-NEXT: smt.int2bv [[V0]] {smt.some_attr} : !smt.bv<4>
  %13 = smt.int2bv %0 {smt.some_attr} : !smt.bv<4>

  return
}
