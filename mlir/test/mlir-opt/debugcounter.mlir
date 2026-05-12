// This test exercises the example in docs/ActionTracing.md ; changes here
// should probably be reflected there.

// RUN: mlir-opt %s -mlir-debug-counter=unique-tag-for-my-action-skip=-1 -mlir-print-debug-counter --pass-pipeline="builtin.module(func.func(canonicalize))" --mlir-disable-threading 2>&1 | FileCheck %s --check-prefix=CHECK-UNKNOWN-TAG
// RUN: mlir-opt %s -mlir-debug-counter=pass-execution-skip=1 -mlir-print-debug-counter --pass-pipeline="builtin.module(func.func(canonicalize))" --mlir-disable-threading 2>&1 | FileCheck %s --check-prefix=CHECK-PASS
// RUN: mlir-opt %s -mlir-debug-counter=pass-execution-skip=1 -test-dialect-conversion-pdll 2>&1 | FileCheck %s --check-prefix=CHECK-PDL-SKIP

func.func @foo() {
    return
}

// CHECK-UNKNOWN-TAG:  DebugCounter counters:
// CHECK-UNKNOWN-TAG: unique-tag-for-my-action        : {0,-1,-1}

// CHECK-PASS: DebugCounter counters:
// CHECK-PASS: pass-execution                  : {1,1,-1}

// Regression test for https://github.com/llvm/llvm-project/issues/131441
// When --mlir-debug-counter skips the internal PDL lowering pass, the
// FrozenRewritePatternSet should handle it gracefully (no crash).
// CHECK-PDL-SKIP-LABEL: func @foo
