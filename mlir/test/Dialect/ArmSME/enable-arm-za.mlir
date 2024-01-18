// RUN: mlir-opt %s -enable-arm-streaming=za-mode=new-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=ENABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=DISABLE-ZA
// RUN: mlir-opt %s -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=NO-ARM-STREAMING

// CHECK-LABEL: @declaration
func.func private @declaration()

// ENABLE-ZA-LABEL: @arm_new_za
// ENABLE-ZA-SAME: attributes {arm_new_za, arm_streaming}
// DISABLE-ZA-LABEL: @arm_new_za
// DISABLE-ZA-NOT: arm_new_za
// DISABLE-ZA-SAME: attributes {arm_streaming}
// NO-ARM-STREAMING-LABEL: @arm_new_za
// NO-ARM-STREAMING-NOT: arm_new_za
// NO-ARM-STREAMING-NOT: arm_streaming
func.func @arm_new_za() { return }
