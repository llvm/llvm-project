// RUN: mlir-opt %s -enable-arm-streaming -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -enable-arm-streaming=streaming-mode=streaming-locally -verify-diagnostics | FileCheck %s -check-prefix=CHECK-LOCALLY
// RUN: mlir-opt %s -enable-arm-streaming=streaming-mode=streaming-compatible -verify-diagnostics | FileCheck %s -check-prefix=CHECK-COMPATIBLE
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=new-za -verify-diagnostics | FileCheck %s -check-prefix=CHECK-ENABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming=if-required-by-ops -verify-diagnostics | FileCheck %s -check-prefix=IF-REQUIRED
// RUN: mlir-opt %s -enable-arm-streaming=if-contains-scalable-vectors -verify-diagnostics | FileCheck %s -check-prefix=IF-SCALABLE

// CHECK-LABEL: @arm_streaming
// CHECK-SAME: attributes {arm_streaming}
// CHECK-LOCALLY-LABEL: @arm_streaming
// CHECK-LOCALLY-SAME: attributes {arm_locally_streaming}
// CHECK-COMPATIBLE-LABEL: @arm_streaming
// CHECK-COMPATIBLE-SAME: attributes {arm_streaming_compatible}
// CHECK-ENABLE-ZA-LABEL: @arm_streaming
// CHECK-ENABLE-ZA-SAME: attributes {arm_new_za, arm_streaming}
func.func @arm_streaming() { return }

// CHECK-LABEL: @not_arm_streaming
// CHECK-SAME: attributes {enable_arm_streaming_ignore}
// CHECK-LOCALLY-LABEL: @not_arm_streaming
// CHECK-LOCALLY-SAME: attributes {enable_arm_streaming_ignore}
// CHECK-COMPATIBLE-LABEL: @not_arm_streaming
// CHECK-COMPATIBLE-SAME: attributes {enable_arm_streaming_ignore}
// CHECK-ENABLE-ZA-LABEL: @not_arm_streaming
// CHECK-ENABLE-ZA-SAME: attributes {enable_arm_streaming_ignore}
func.func @not_arm_streaming() attributes {enable_arm_streaming_ignore} { return }

// CHECK-LABEL: @requires_arm_streaming
// CHECK-SAME: attributes {arm_streaming}
// IF-REQUIRED: @requires_arm_streaming
// IF-REQUIRED-SAME: attributes {arm_streaming}
func.func @requires_arm_streaming() {
  %tile = arm_sme.get_tile : vector<[4]x[4]xi32>
  return
}

// CHECK-LABEL: @does_not_require_arm_streaming
// CHECK-SAME: attributes {arm_streaming}
// IF-REQUIRED: @does_not_require_arm_streaming
// IF-REQUIRED-NOT: arm_streaming
func.func @does_not_require_arm_streaming() { return }

// IF-SCALABLE-LABEL: @contains_scalable_vectors
// IF-SCALABLE-SAME: attributes {arm_streaming}
func.func @contains_scalable_vectors(%vec: vector<[4]xf32>) -> vector<[4]xf32> {
  %0 = arith.addf %vec, %vec : vector<[4]xf32>
  return %0 : vector<[4]xf32>
}

// IF-SCALABLE-LABEL: @no_scalable_vectors
// IF-SCALABLE-NOT: arm_streaming
func.func @no_scalable_vectors(%vec: vector<4xf32>) -> vector<4xf32> {
  %0 = arith.addf %vec, %vec : vector<4xf32>
  return %0 : vector<4xf32>
}
