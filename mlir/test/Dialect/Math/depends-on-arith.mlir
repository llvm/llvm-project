// RUN: mlir-opt %s --mlir-print-op-generic | FileCheck %s

// Check that math.atan can be constructed by parsing and the fastmath
// attribute can be created. This requires math dialect to depend on arith
// dialect. Note that we don't want other dialects in here as they may
// transitively depend on arith and load it even if math doesn't.

"test.some_op_with_region"() ({
^bb0(%arg0: f64):
  // CHECK: #arith.fastmath<none>
  math.atan %arg0 : f64
  "test.possible_terminator"() : () -> ()
}) : () -> ()
