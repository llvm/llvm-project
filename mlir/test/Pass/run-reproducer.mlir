// RUN: mlir-opt %s --run-reproducer -dump-pass-pipeline 2>&1 | FileCheck %s
// RUN: mlir-opt %s --run-reproducer -mlir-print-ir-before=cse 2>&1 | FileCheck -check-prefix=BEFORE %s

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
      verify_each: true,
      // CHECK:  builtin.module(func.func(cse,canonicalize{ max-iterations=1 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=false}))
      pipeline: "builtin.module(func.func(cse,canonicalize{max-iterations=1 max-num-rewrites=-1 region-simplify=normal top-down=false}))",
      disable_threading: true
    }
  }
#-}

// BEFORE: // -----// IR Dump Before{{.*}}CSE (cse) //----- //
// BEFORE-NEXT: func @foo()
// BEFORE: // -----// IR Dump Before{{.*}}CSE (cse) //----- //
// BEFORE-NEXT: func @bar()
// BEFORE-NOT: // -----// IR Dump Before{{.*}}Canonicalizer (canonicalize) //----- //
// BEFORE-NOT: // -----// IR Dump After
