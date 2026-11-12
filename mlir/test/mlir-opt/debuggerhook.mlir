// Checks that the debugger hook is enabled when called with the CLI option.
// RUN: mlir-opt %s --mlir-enable-debugger-hook --pass-pipeline="builtin.module(func.func(canonicalize))" --mlir-disable-threading 2>&1 | FileCheck %s

func.func @foo() {
    return
}

// CHECK: ExecutionContext registered on the context
// CHECK-SAME:  (with Debugger hook)
