// RUN: mlir-opt %s -enable-arm-streaming=za-mode=new-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=ENABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=DISABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=in-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=IN-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=out-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=OUT-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=inout-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=INOUT-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=preserves-za -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=PRESERVES-ZA
// RUN: mlir-opt %s -convert-arm-sme-to-llvm | FileCheck %s -check-prefix=NO-ARM-STREAMING

// CHECK-LABEL: @declaration
func.func private @declaration()

// ENABLE-ZA-LABEL: @arm_new_za
// ENABLE-ZA-SAME: attributes {arm_new_za, arm_streaming}
// IN-ZA-LABEL: @arm_new_za
// IN-ZA-SAME: attributes {arm_in_za, arm_streaming}
// OUT-ZA-LABEL: @arm_new_za
// OUT-ZA-SAME: attributes {arm_out_za, arm_streaming}
// INOUT-ZA-LABEL: @arm_new_za
// INOUT-ZA-SAME: attributes {arm_inout_za, arm_streaming}
// PRESERVES-ZA-LABEL: @arm_new_za
// PRESERVES-ZA-SAME: attributes {arm_preserves_za, arm_streaming}
// DISABLE-ZA-LABEL: @arm_new_za
// DISABLE-ZA-NOT: arm_new_za
// DISABLE-ZA-SAME: attributes {arm_streaming}
// NO-ARM-STREAMING-LABEL: @arm_new_za
// NO-ARM-STREAMING-NOT: arm_new_za
// NO-ARM-STREAMING-NOT: arm_streaming
// NO-ARM-STREAMING-NOT: arm_in_za
// NO-ARM-STREAMING-NOT: arm_out_za
// NO-ARM-STREAMING-NOT: arm_inout_za
// NO-ARM-STREAMING-NOT: arm_preserves_za
func.func @arm_new_za() { return }
