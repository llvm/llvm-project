// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='func.func(cse,canonicalize)' -mlir-print-ir-before=cse  -o /dev/null 2>&1 | FileCheck -check-prefix=BEFORE %s
// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='func.func(cse,canonicalize)' -mlir-print-ir-before-all -o /dev/null 2>&1 | FileCheck -check-prefix=BEFORE_ALL %s
// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='func.func(cse,canonicalize)' -mlir-print-ir-after=cse -o /dev/null 2>&1 | FileCheck -check-prefix=AFTER %s
// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='func.func(cse,canonicalize)' -mlir-print-ir-after-all -o /dev/null 2>&1 | FileCheck -check-prefix=AFTER_ALL %s
// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='func.func(cse,canonicalize)' -mlir-print-ir-before=cse -mlir-print-ir-module-scope -o /dev/null 2>&1 | FileCheck -check-prefix=BEFORE_MODULE %s
// RUN: mlir-opt %s -mlir-disable-threading=true -pass-pipeline='func.func(cse,cse)' -mlir-print-ir-after-all -mlir-print-ir-after-change -o /dev/null 2>&1 | FileCheck -check-prefix=AFTER_ALL_CHANGE %s
// RUN: not mlir-opt %s -mlir-disable-threading=true -pass-pipeline='func.func(cse,test-pass-failure)' -mlir-print-ir-after-failure -o /dev/null 2>&1 | FileCheck -check-prefix=AFTER_FAILURE %s

func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

func.func @bar() {
  return
}

// BEFORE: // -----// IR Dump Before{{.*}}CSE (cse) //----- //
// BEFORE-NEXT: func @foo()
// BEFORE: // -----// IR Dump Before{{.*}}CSE (cse) //----- //
// BEFORE-NEXT: func @bar()
// BEFORE-NOT: // -----// IR Dump Before{{.*}}Canonicalizer (canonicalize) //----- //
// BEFORE-NOT: // -----// IR Dump After

// BEFORE_ALL: // -----// IR Dump Before{{.*}}CSE (cse) //----- //
// BEFORE_ALL-NEXT: func @foo()
// BEFORE_ALL: // -----// IR Dump Before{{.*}}Canonicalizer (canonicalize) //----- //
// BEFORE_ALL-NEXT: func @foo()
// BEFORE_ALL: // -----// IR Dump Before{{.*}}CSE (cse) //----- //
// BEFORE_ALL-NEXT: func @bar()
// BEFORE_ALL: // -----// IR Dump Before{{.*}}Canonicalizer (canonicalize) //----- //
// BEFORE_ALL-NEXT: func @bar()
// BEFORE_ALL-NOT: // -----// IR Dump After

// AFTER-NOT: // -----// IR Dump Before
// AFTER: // -----// IR Dump After{{.*}}CSE (cse) //----- //
// AFTER-NEXT: func @foo()
// AFTER: // -----// IR Dump After{{.*}}CSE (cse) //----- //
// AFTER-NEXT: func @bar()
// AFTER-NOT: // -----// IR Dump After{{.*}}Canonicalizer (canonicalize) //----- //

// AFTER_ALL-NOT: // -----// IR Dump Before
// AFTER_ALL: // -----// IR Dump After{{.*}}CSE (cse) //----- //
// AFTER_ALL-NEXT: func @foo()
// AFTER_ALL: // -----// IR Dump After{{.*}}Canonicalizer (canonicalize) //----- //
// AFTER_ALL-NEXT: func @foo()
// AFTER_ALL: // -----// IR Dump After{{.*}}CSE (cse) //----- //
// AFTER_ALL-NEXT: func @bar()
// AFTER_ALL: // -----// IR Dump After{{.*}}Canonicalizer (canonicalize) //----- //
// AFTER_ALL-NEXT: func @bar()

// BEFORE_MODULE: // -----// IR Dump Before{{.*}}CSE (cse) ('func.func' operation: @foo) //----- //
// BEFORE_MODULE: func @foo()
// BEFORE_MODULE: func @bar()
// BEFORE_MODULE: // -----// IR Dump Before{{.*}}CSE (cse) ('func.func' operation: @bar) //----- //
// BEFORE_MODULE: func @foo()
// BEFORE_MODULE: func @bar()

// AFTER_ALL_CHANGE: // -----// IR Dump After{{.*}}CSE (cse) //----- //
// AFTER_ALL_CHANGE-NEXT: func @foo()
// AFTER_ALL_CHANGE-NOT: // -----// IR Dump After{{.*}}CSE (cse) //----- //
// We expect that only 'foo' changed during CSE, and the second run of CSE did
// nothing.

// AFTER_FAILURE-NOT: // -----// IR Dump After{{.*}}CSE
// AFTER_FAILURE: // -----// IR Dump After{{.*}}TestFailurePass Failed (test-pass-failure) //----- //
// AFTER_FAILURE: func @foo()
