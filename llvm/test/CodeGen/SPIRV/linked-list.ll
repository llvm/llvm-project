; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

%struct.Node = type { ptr addrspace(1) }
%struct.Node.0 = type opaque

; With opaque pointers, parameters are lowered to generic pointer-to-i8.
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I8]]
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32
; CHECK-DAG: %[[#VOID:]] = OpTypeVoid
; CHECK-DAG: %[[#FNTY:]] = OpTypeFunction %[[#VOID]] %[[#PTR]] %[[#PTR]] %[[#I32]]
; CHECK:     %[[#]] = OpFunction %[[#VOID]] None %[[#FNTY]]
; CHECK:     OpFunctionParameter %[[#PTR]]
; CHECK:     OpFunctionParameter %[[#PTR]]
; CHECK:     OpFunctionParameter %[[#I32]]

define spir_kernel void @create_linked_lists(ptr addrspace(1) nocapture %pNodes, ptr addrspace(1) nocapture %allocation_index, i32 %list_length) {
entry:
  ret void
}
