// RUN: mlir-opt -test-print-defuse  -allow-unregistered-dialect %s | FileCheck %s

// CHECK: Visiting op 'dialect.op1' with 0 operands:
// CHECK: Has 4 results:
// CHECK:   - Result 0 has a single use:     - dialect.op2
// CHECK:   - Result 1 has no uses
// CHECK:   - Result 2 has 2 uses:
// CHECK:     - dialect.innerop1
// CHECK:     - dialect.op2
// CHECK:   - Result 3 has no uses
// CHECK: Visiting op 'dialect.op2' with 2 operands:
// CHECK:   - Operand produced by operation 'dialect.op1'
// CHECK:   - Operand produced by operation 'dialect.op1'
// CHECK: Has 0 results:
// CHECK: Visiting op 'dialect.innerop1' with 2 operands:
// CHECK:   - Operand produced by Block argument, number 0
// CHECK:   - Operand produced by operation 'dialect.op1'
// CHECK: Has 0 results:
// CHECK: Visiting op 'dialect.op3' with 0 operands:
// CHECK: Has 0 results:
// CHECK: Visiting op 'module_terminator' with 0 operands:
// CHECK: Has 0 results:
// CHECK: Visiting op 'module' with 0 operands:
// CHECK: Has 0 results:

%results:4 = "dialect.op1"() : () -> (i1, i16, i32, i64)
"dialect.op2"(%results#0, %results#2) : (i1, i32) -> ()
"dialect.op3"() ({
  ^bb0(%arg0 : i1):
    "dialect.innerop1"(%arg0, %results#2) : (i1, i32) -> ()
}) : () -> ()
