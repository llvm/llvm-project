// RUN: mlir-opt -test-transform-dialect-interpreter %s -allow-unregistered-dialect -verify-diagnostics --split-input-file | FileCheck %s

// CHECK: func.func @foo() {
// CHECK:   "dummy_op"() : () -> ()
// CHECK: }
// CHECK-NOT: func.func @bar
func.func @bar() {
  "another_op"() : () -> ()
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.replace %0 {
    func.func @foo() {
      "dummy_op"() : () -> ()
    }
  }
}

// -----

func.func @bar(%arg0: i1) {
  "another_op"(%arg0) : (i1) -> ()
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["another_op"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @+1 {{expected target without operands}}
  transform.structured.replace %0 {
    "dummy_op"() : () -> ()
  }
}

// -----

func.func @bar() {
  "another_op"() : () -> ()
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["another_op"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.structured.replace %0 {
  ^bb0(%a: i1):
    // expected-error @+1 {{expected replacement without operands}}
    "dummy_op"(%a) : (i1) -> ()
  }
}
