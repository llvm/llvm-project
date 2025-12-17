// RUN: mlir-opt %s | FileCheck %s

// Check that attr and type aliases are properly printed.

// CHECK: {types = [!test.tuple_i7_from_attr, !test.tuple_i6_from_attr, tuple<!test.int<signed, 5>>]} : () -> (!test.tuple_i7, !test.tuple_i6, tuple<!test.int<signed, 5>>)
"test.op"() {types = [
    tuple<!test.int<s, 7>>,
    tuple<!test.int<s, 6>>,
    tuple<!test.int<s, 5>>
  ]} : () -> (
    tuple<!test.int<s, 7>>,
    tuple<!test.int<s, 6>>,
    tuple<!test.int<s, 5>>
  )
