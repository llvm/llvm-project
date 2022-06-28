// RUN: mlir-opt %s -mlir-print-ir-before=cse 2>&1 | FileCheck -check-prefix=BEFORE %s

func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

func.func @bar() {
  return
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "func.func(cse,canonicalize)",
      disable_threading: true
    }
  }
#-}

// BEFORE: // -----// IR Dump Before{{.*}}CSE //----- //
// BEFORE-NEXT: func @foo()
// BEFORE: // -----// IR Dump Before{{.*}}CSE //----- //
// BEFORE-NEXT: func @bar()
// BEFORE-NOT: // -----// IR Dump Before{{.*}}Canonicalizer //----- //
// BEFORE-NOT: // -----// IR Dump After
