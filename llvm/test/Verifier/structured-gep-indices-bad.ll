; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

%S = type { i32, i32 }

define void @too_many_indices(ptr %src) {
entry:
; CHECK: Reached a non-composite type with more indices to process
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 0, i32 0)
  ret void
}

define void @out_of_bounds_struct_access(ptr %src) {
entry:
; CHECK: Indexing in a struct should be inbounds
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 2)
  ret void
}

define void @dynamic_index_struct(ptr %src, i32 %index) {
entry:
; CHECK: Indexing into a struct requires a constant int
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%S) %src, i32 %index)
  ret void
}

define void @non_integer_operand(ptr %src) {
entry:
; CHECK: Index operand type must be an integer
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([ 2 x i32 ]) %src, float 1.0)
  ret void
}

define void @missing_attribute(ptr %src) {
entry:
; CHECK: Intrinsic first parameter is missing an ElementType attribute
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr %src, i32 0)
  ret void
}

define void @amdgpu_stridemark_offset_in_structured_gep(ptr addrspace(9) %src) {
entry:
; CHECK: Reached a non-composite type with more indices to process
  %ptr = call ptr addrspace(9) (ptr addrspace(9), ...) @llvm.structured.gep.p9(ptr addrspace(9) elementtype([0 x target("amdgpu.stridemark")]) %src, i32 0, i32 0)
  ret void
}
