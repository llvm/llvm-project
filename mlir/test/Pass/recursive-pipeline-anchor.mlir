// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline='builtin.module(**func.func(test-function-pass))' -verify-each=false -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 | FileCheck %s
// RUN: mlir-opt %s -mlir-disable-threading -test-function-pass --pass-pipeline-anchor=func.func -verify-each=false -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 | FileCheck %s

// some with threading enabled

// RUN: mlir-opt %s -pass-pipeline='builtin.module(**func.func(test-function-pass))' -verify-each=false -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 | FileCheck %s
// RUN: mlir-opt %s -test-function-pass --pass-pipeline-anchor=func.func -verify-each=false -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 | FileCheck %s

// some without recursion

// RUN: mlir-opt %s -mlir-disable-threading -test-function-pass -verify-each=false -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 | FileCheck %s --check-prefix=NON_REC_CHECK
// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline='builtin.module(func.func(test-function-pass))' -verify-each=false -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 | FileCheck %s --check-prefix=NON_REC_CHECK

func.func @foo() {
  return
}

module {
  func.func @bar() {
    return
  }
}

// with recursive anchor the pass is executed on @foo and @bar

// CHECK:                TestFunctionPass
// CHECK-NEXT:              (S) 2 counter - Number of invocations

// in non-recursive mode the pass is only executed on @foo

// NON_REC_CHECK:        TestFunctionPass
// NON_REC_CHECK-NEXT:      (S) 1 counter - Number of invocations