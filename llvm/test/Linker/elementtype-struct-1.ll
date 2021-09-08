; RUN: llvm-link %S/Inputs/elementtype-struct-2.ll %s -S | FileCheck %s

; Check that the attribute for elementtype matches when linking.

; CHECK: define void @struct_elementtype_2
; CHECK: call %struct* @llvm.preserve.array.access.index.p0s_structs.p0s_structs(%struct* elementtype(%struct) null, i32 0, i32 0)
; CHECK: define void @struct_elementtype
; CHECK: call %struct* @llvm.preserve.array.access.index.p0s_structs.p0s_structs(%struct* elementtype(%struct) null, i32 0, i32 0)

%struct = type {i32, i8}

define void @struct_elementtype() {
  call %struct* @llvm.preserve.array.access.index.p0s_structs.p0s_structs(%struct* elementtype(%struct) null, i32 0, i32 0)
  ret void
}

declare %struct* @llvm.preserve.array.access.index.p0s_structs.p0s_structs(%struct*, i32, i32)
