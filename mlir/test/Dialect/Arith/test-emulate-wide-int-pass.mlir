// Check that the test version of the wide integer emulation pass applies
// conversion to functions whose name start with a given prefix only, and that
// the function signatures are preserved.

// RUN: mlir-opt %s --test-arith-emulate-wide-int="function-prefix=emulate_me_" | FileCheck %s

// CHECK-LABEL: func.func @entry()
// CHECK:         {{%.+}} = call @emulate_me_please({{.+}}) : (i64) -> i64
// CHECK-NEXT:    {{%.+}} = call @foo({{.+}}) : (i64) -> i64
func.func @entry() {
  %cst0 = arith.constant 0 : i64
  func.call @emulate_me_please(%cst0) : (i64) -> (i64)
  func.call @foo(%cst0) : (i64) -> (i64)
  return
}

// CHECK-LABEL: func.func @emulate_me_please
// CHECK-SAME:    ([[ARG:%.+]]: i64) -> i64 {
// CHECK-NEXT:    [[BCAST0:%.+]]   = llvm.bitcast [[ARG]] : i64 to vector<2xi32>
// CHECK-NEXT:    [[LOW0:%.+]]     = vector.extract [[BCAST0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH0:%.+]]    = vector.extract [[BCAST0]][1] : vector<2xi32>
// CHECK-NEXT:    [[LOW1:%.+]]     = vector.extract [[BCAST0]][0] : vector<2xi32>
// CHECK-NEXT:    [[HIGH1:%.+]]    = vector.extract [[BCAST0]][1] : vector<2xi32>
// CHECK-NEXT:    {{%.+}}, {{%.+}} = arith.addui_carry [[LOW0]], [[LOW1]] : i32, i1
// CHECK:         [[RES:%.+]]      = llvm.bitcast {{%.+}} : vector<2xi32> to i64
// CHECK-NEXt:    return [[RES]] : i64
func.func @emulate_me_please(%x : i64) -> i64 {
  %r = arith.addi %x, %x : i64
  return %r : i64
}

// CHECK-LABEL: func.func @foo
// CHECK-SAME:    ([[ARG:%.+]]: i64) -> i64 {
// CHECK-NEXT:    [[RES:%.+]] = arith.addi [[ARG]], [[ARG]] : i64
// CHECK-NEXT:    return [[RES]] : i64
func.func @foo(%x : i64) -> i64 {
  %r = arith.addi %x, %x : i64
  return %r : i64
}
