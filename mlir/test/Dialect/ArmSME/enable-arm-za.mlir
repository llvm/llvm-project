// RUN: mlir-opt %s -enable-arm-streaming=za-mode=new-za | FileCheck %s -check-prefix=ENABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming | FileCheck %s -check-prefix=DISABLE-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=in-za | FileCheck %s -check-prefix=IN-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=out-za | FileCheck %s -check-prefix=OUT-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=inout-za | FileCheck %s -check-prefix=INOUT-ZA
// RUN: mlir-opt %s -enable-arm-streaming=za-mode=preserves-za | FileCheck %s -check-prefix=PRESERVES-ZA

// CHECK-LABEL: @declaration
func.func private @declaration()

// ENABLE-ZA-LABEL: @arm_new_za
// ENABLE-ZA-SAME: attributes {llvm.arm_new_za, llvm.arm_streaming}
// IN-ZA-LABEL: @arm_new_za
// IN-ZA-SAME: attributes {llvm.arm_in_za, llvm.arm_streaming}
// OUT-ZA-LABEL: @arm_new_za
// OUT-ZA-SAME: attributes {llvm.arm_out_za, llvm.arm_streaming}
// INOUT-ZA-LABEL: @arm_new_za
// INOUT-ZA-SAME: attributes {llvm.arm_inout_za, llvm.arm_streaming}
// PRESERVES-ZA-LABEL: @arm_new_za
// PRESERVES-ZA-SAME: attributes {llvm.arm_preserves_za, llvm.arm_streaming}
// DISABLE-ZA-LABEL: @arm_new_za
// DISABLE-ZA-NOT: arm_new_za
// DISABLE-ZA-SAME: attributes {llvm.arm_streaming}
func.func @arm_new_za() { return }
