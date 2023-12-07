// This test exercises the example in docs/ActionTracing.md ; changes here
// should probably be reflected there.

// RUN: mlir-opt %s -mlir-debug-counter=unique-tag-for-my-action-skip=-1 -mlir-print-debug-counter --pass-pipeline="builtin.module(func.func(canonicalize))" --mlir-disable-threading 2>&1 | FileCheck %s --check-prefix=CHECK-UNKNOWN-TAG
// RUN: mlir-opt %s -mlir-debug-counter=pass-execution-skip=1 -mlir-print-debug-counter --pass-pipeline="builtin.module(func.func(canonicalize))" --mlir-disable-threading 2>&1 | FileCheck %s --check-prefix=CHECK-PASS

func.func @foo() {
    return
}

// CHECK-UNKNOWN-TAG:  DebugCounter counters:
// CHECK-UNKNOWN-TAG: unique-tag-for-my-action        : {0,-1,-1}

// CHECK-PASS: DebugCounter counters:
// CHECK-PASS: pass-execution                  : {1,1,-1}
