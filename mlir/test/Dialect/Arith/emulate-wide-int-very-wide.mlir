// Check that emulation of wery wide types (>64 bits) works as expected.

// RUN: mlir-opt --arith-emulate-wide-int="widest-int-supported=512" %s | FileCheck %s

// CHECK-LABEL: func.func @muli_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi512>, [[ARG1:%.+]]: vector<2xi512>) -> vector<2xi512>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2xi512>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2xi512>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2xi512>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2xi512>
//
// CHECK-DAG:     arith.mului_extended
// CHECK-DAG:     arith.muli
// CHECK-DAG:     arith.muli
// CHECK-NEXT:    arith.addi
// CHECK-NEXT:    arith.addi
//
// CHECK:         return {{%.+}} : vector<2xi512>
func.func @muli_scalar(%a : i1024, %b : i1024) -> i1024 {
    %m = arith.muli %a, %b : i1024
    return %m : i1024
}
