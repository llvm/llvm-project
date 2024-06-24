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

// CHECK: "test.test_array_float"
// CHECK-SAME: 1.000000e+00 : f32, 1.000000e+00, 0x7FF0000000000000 : f64
"test.test_array_float"() {test.float_arr = [1.0 : f32, 1.0 : f64, 0x7FF0000000000000 : f64]} : () -> ()
