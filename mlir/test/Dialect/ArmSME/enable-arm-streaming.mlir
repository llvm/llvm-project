// RUN: mlir-opt %s -enable-arm-streaming -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -enable-arm-streaming=mode=locally -verify-diagnostics | FileCheck %s -check-prefix=CHECK-LOCALLY
// RUN: mlir-opt %s -enable-arm-streaming=enable-za -verify-diagnostics | FileCheck %s -check-prefix=CHECK-ENABLE-ZA

// CHECK-LABEL: @arm_streaming
// CHECK-SAME: attributes {arm_streaming}
// CHECK-LOCALLY-LABEL: @arm_streaming
// CHECK-LOCALLY-SAME: attributes {arm_locally_streaming}
// CHECK-ENABLE-ZA-LABEL: @arm_streaming
// CHECK-ENABLE-ZA-SAME: attributes {arm_streaming, arm_za}
func.func @arm_streaming() { return }
