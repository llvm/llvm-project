// RUN: mlir-opt -test-print-nesting  -allow-unregistered-dialect %s | FileCheck %s

// CHECK: visiting op: 'builtin.module' with 0 operands and 0 results
// CHECK:  1 nested regions:
// CHECK:   Region with 1 blocks:
// CHECK:     Block with 0 arguments, 0 successors, and 2 operations
module {


// CHECK:       visiting op: 'dialect.op1' with 0 operands and 4 results
// CHECK:       1 attributes:
// CHECK:        - 'attribute name' : '42 : i32'
// CHECK:        0 nested regions:
  %results:4 = "dialect.op1"() { "attribute name" = 42 : i32 } : () -> (i1, i16, i32, i64)


// CHECK:       visiting op: 'dialect.op2' with 0 operands and 0 results
// CHECK:        2 nested regions:
  "dialect.op2"() ({

// CHECK:         Region with 1 blocks:
// CHECK:           Block with 0 arguments, 0 successors, and 1 operations
// CHECK:             visiting op: 'dialect.innerop1' with 2 operands and 0 results
// CHECK:              0 nested regions:
    "dialect.innerop1"(%results#0, %results#1) : (i1, i16) -> ()

// CHECK:         Region with 3 blocks:
  },{

// CHECK:           Block with 0 arguments, 2 successors, and 2 operations
// CHECK:             visiting op: 'dialect.innerop2' with 0 operands and 0 results
// CHECK:              0 nested regions:
    "dialect.innerop2"() : () -> ()
// CHECK:             visiting op: 'dialect.innerop3' with 3 operands and 0 results
// CHECK:              0 nested regions:
    "dialect.innerop3"(%results#0, %results#2, %results#3)[^bb1, ^bb2] : (i1, i32, i64) -> ()
// CHECK:           Block with 1 arguments, 0 successors, and 2 operations
  ^bb1(%arg1 : i32):
// CHECK:             visiting op: 'dialect.innerop4' with 0 operands and 0 results
// CHECK:              0 nested regions:
    "dialect.innerop4"() : () -> ()
// CHECK:             visiting op: 'dialect.innerop5' with 0 operands and 0 results
// CHECK:              0 nested regions:
    "dialect.innerop5"() : () -> ()
// CHECK:           Block with 1 arguments, 0 successors, and 2 operations
  ^bb2(%arg2 : i64):
// CHECK:             visiting op: 'dialect.innerop6' with 0 operands and 0 results
// CHECK:              0 nested regions:
    "dialect.innerop6"() : () -> ()
// CHECK:             visiting op: 'dialect.innerop7' with 0 operands and 0 results
// CHECK:              0 nested regions:
    "dialect.innerop7"() : () -> ()
  }) : () -> ()

} // module
