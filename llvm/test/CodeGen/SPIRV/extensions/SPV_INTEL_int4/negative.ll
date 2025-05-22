; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_integers %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-INT-4

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-INT-8
; No error would be reported in comparison to Khronos llvm-spirv, because type adjustments to integer size are made 
; in case no appropriate extension is enabled. Here we expect that the type is adjusted to 8 bits.

; CHECK-SPIRV: Capability ArbitraryPrecisionIntegersINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_arbitrary_precision_integers"
; CHECK-INT-4: %[[#Int4:]] = OpTypeInt 4 0
; CHECK-INT-8: %[[#Int4:]] = OpTypeInt 8 0
; CHECK: OpTypeFunction %[[#]] %[[#Int4]]
; CHECK: %[[#Int4PtrTy:]] = OpTypePointer Function %[[#Int4]]
; CHECK: %[[#Const:]] = OpConstant %[[#Int4]]  1

; CHECK: %[[#Int4Ptr:]] = OpVariable %[[#Int4PtrTy]] Function
; CHECK: OpStore %[[#Int4Ptr]] %[[#Const]]
; CHECK: %[[#Load:]] = OpLoad %[[#Int4]] %[[#Int4Ptr]]
; CHECK: OpFunctionCall %[[#]] %[[#]] %[[#Load]]

define spir_kernel void @foo() {
entry:
  %0 = alloca i4
  store i4 1, ptr %0
  %1 = load i4, ptr %0
  call spir_func void @boo(i4 %1)
  ret void
}

declare spir_func void @boo(i4)
