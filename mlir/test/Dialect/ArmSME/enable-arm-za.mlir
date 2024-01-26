// RUN: mlir-opt %s -enable-arm-streaming=za-mode=new-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=ENABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=DISABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=shared-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=SHARED-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=preserves-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=PRESERVES-ZA
// RUN: mlir-opt %s -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=NO-ARM-STREAMING

// CHECK-LABEL: @declaration
func.func private @declaration()

// ENABLE-ZA-LABEL: @arm_new_za
// ENABLE-ZA-SAME: attributes {arm_new_za, arm_streaming}
// SHARED-ZA-LABEL: @arm_new_za
// SHARED-ZA-SAME: attributes {arm_shared_za, arm_streaming}
// PRESERVES-ZA-LABEL: @arm_new_za
// PRESERVES-ZA-SAME: attributes {arm_preserves_za, arm_streaming}
// DISABLE-ZA-LABEL: @arm_new_za
// DISABLE-ZA-NOT: arm_new_za
// DISABLE-ZA-SAME: attributes {arm_streaming}
// NO-ARM-STREAMING-LABEL: @arm_new_za
// NO-ARM-STREAMING-NOT: arm_new_za
// NO-ARM-STREAMING-NOT: arm_streaming
// NO-ARM-STREAMING-NOT: arm_shared_za
// NO-ARM-STREAMING-NOT: arm_preserves_za
func.func @arm_new_za() { return }
