// RUN: mlir-opt %s -enable-arm-streaming -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -enable-arm-streaming=mode=locally -verify-diagnostics | FileCheck %s -check-prefix=CHECK-LOCALLY

// CHECK-LABEL: @arm_streaming
// CHECK-SAME: attributes {arm_streaming}
// CHECK-LOCALLY-LABEL: @arm_streaming
// CHECK-LOCALLY-SAME: attributes {arm_locally_streaming}
func.func @arm_streaming() { return }
