// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK: test.array_of_attr_op
test.array_of_attr_op
    // CHECK-SAME: a = [ begin 0 : index end, begin 2 : index end ]
    a = [begin 0 : index end, begin 2 : index end],
    // CHECK-SAME: [0, 1, -42, 42]
    b = [0, 1, -42, 42],
    // CHECK-SAME: [a, b, b, a]
    c = [a, b, b, a]

// CHECK: test.array_of_attr_op
// CHECK-SAME: a = [], b = [], c = []
test.array_of_attr_op a = [], b = [], c = []
