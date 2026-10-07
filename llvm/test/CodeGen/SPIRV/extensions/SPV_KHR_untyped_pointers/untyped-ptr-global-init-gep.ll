; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s

; No spirv-val run here. SPIRV-Tools rejects opcode 4424 inside OpSpecConstantOp
; while parsing, because its table of valid opcodes predates this extension.

; A constant expression GEP over an untyped global needs
; OpUntypedInBoundsPtrAccessChainKHR, which spells out its Base Type. The offset
; is a byte count, so the Base Type is i8 and the offset is the Element index as
; is. Here element 2 of a [4 x i32] is byte 8.

; CHECK-DAG: OpCapability UntypedPointersKHR
; CHECK-DAG: OpExtension "SPV_KHR_untyped_pointers"

; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#PTR_CW:]] = OpTypeUntypedPointerKHR CrossWorkgroup
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#I32]] %[[#]]
; CHECK-DAG: %[[#OFF:]] = OpConstant %[[#I64]] 8

; CHECK-DAG: %[[#SRC:]] = OpUntypedVariableKHR %[[#PTR_CW]] CrossWorkgroup %[[#ARR]] %[[#]]
; CHECK-DAG: %[[#GEP:]] = OpSpecConstantOp %[[#PTR_CW]] UntypedInBoundsPtrAccessChainKHR %[[#I8]] %[[#SRC]] %[[#OFF]]
; CHECK-DAG: %[[#]] = OpUntypedVariableKHR %[[#PTR_CW]] CrossWorkgroup %[[#PTR_CW]] %[[#GEP]]

@src = addrspace(1) global [4 x i32] zeroinitializer
@p = addrspace(1) global ptr addrspace(1) getelementptr inbounds ([4 x i32], ptr addrspace(1) @src, i32 0, i32 2)
