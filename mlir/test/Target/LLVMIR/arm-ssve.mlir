// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// Attribute to enable streaming-mode.

// CHECK-LABEL: @streaming_callee
// CHECK: #[[ATTR:[0-9]*]]
llvm.func @streaming_callee() attributes {passthrough = ["aarch64_pstate_sm_enabled"]} {
  llvm.return
}

// CHECK: attributes #[[ATTR]] = { "aarch64_pstate_sm_enabled" }
