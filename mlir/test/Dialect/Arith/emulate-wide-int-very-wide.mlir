// Check that emulation of wery wide types (>64 bits) works as expected.

// RUN: mlir-opt --arith-emulate-wide-int="widest-int-supported=512" %s | FileCheck %s

// CHECK-LABEL: func.func @muli_scalar
// CHECK-SAME:    ([[ARG0:%.+]]: vector<2xi512>, [[ARG1:%.+]]: vector<2xi512>) -> vector<2xi512>
// CHECK-NEXT:    [[LOW0:%.+]]   = vector.extract [[ARG0]][0] : vector<2xi512>
// CHECK-NEXT:    [[HIGH0:%.+]]  = vector.extract [[ARG0]][1] : vector<2xi512>
// CHECK-NEXT:    [[LOW1:%.+]]   = vector.extract [[ARG1]][0] : vector<2xi512>
// CHECK-NEXT:    [[HIGH1:%.+]]  = vector.extract [[ARG1]][1] : vector<2xi512>
//
// Check that the mask for the low 256-bits was generated correctly. The exact expected value is 2^256 - 1.
// CHECK-NEXT:    {{.+}}         = arith.constant 115792089237316195423570985008687907853269984665640564039457584007913129639935 : i512
// CHECK:         return {{%.+}} : vector<2xi512>
func.func @muli_scalar(%a : i1024, %b : i1024) -> i1024 {
    %m = arith.muli %a, %b : i1024
    return %m : i1024
}
