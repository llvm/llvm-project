// RUN: mlir-opt %s -enable-arm-streaming=enable-za -convert-vector-to-llvm="enable-arm-sme" | FileCheck %s -check-prefix=ENABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming -convert-vector-to-llvm="enable-arm-sme" | FileCheck %s -check-prefix=DISABLE-ZA
// RUN: mlir-opt %s -convert-vector-to-llvm="enable-arm-sme" | FileCheck %s -check-prefix=NO-ARM-STREAMING

// CHECK-LABEL: @declaration
func.func private @declaration()

// CHECK-LABEL: @arm_za
func.func @arm_za() {
  // ENABLE-ZA: arm_sme.intr.za.enable
  // ENABLE-ZA-NEXT: arm_sme.intr.za.disable
  // ENABLE-ZA-NEXT: return
  // DISABLE-ZA-NOT: arm_sme.intr.za.enable
  // DISABLE-ZA-NOT: arm_sme.intr.za.disable
  // NO-ARM-STREAMING-NOT: arm_sme.intr.za.enable
  // NO-ARM-STREAMING-NOT: arm_sme.intr.za.disable
  return
}
