; RUN: llvm-link %S/Inputs/elementtype-struct-2.ll %s -S | FileCheck %s

; Check that the attribute for elementtype matches when linking.

; CHECK: define void @struct_elementtype_2
; CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(%struct) null, i32 0, i32 0)
; CHECK: define void @struct_elementtype
; CHECK: call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(%struct) null, i32 0, i32 0)

%struct = type {i32, i8}

define void @struct_elementtype() {
  call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype(%struct) null, i32 0, i32 0)
  ret void
}

declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32, i32)
