// RUN: mlir-opt %s -verify-diagnostics -pass-pipeline='any(cse, test-interface-pass)' -allow-unregistered-dialect -o /dev/null

// Test that we execute generic pipelines correctly. The `cse` pass is fully generic and should execute
// on both the module and the func. The `test-interface-pass` filters based on FunctionOpInterface and
// should only execute on the func.

// expected-remark@below {{Executing interface pass on operation}}
func.func @main() -> (i1, i1) {
  // CHECK-LABEL: func @main
  // CHECK-NEXT: arith.constant true
  // CHECK-NEXT: return
  %true = arith.constant true
  %true1 = arith.constant true
  return %true, %true1 : i1, i1
}

module @module {
  // CHECK-LABEL: module @main
  // CHECK-NEXT: arith.constant true
  // CHECK-NEXT: foo.op
  %true = arith.constant true
  %true1 = arith.constant true
  "foo.op"(%true, %true1) : (i1, i1) -> ()
}
