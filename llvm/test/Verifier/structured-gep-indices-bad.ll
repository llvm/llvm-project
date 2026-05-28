; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

%S = type { i32, i32 }

define void @too_many_indices(ptr %src) {
entry:
; CHECK: Reached a non-composite type with more indices to process
  %ptr = call ptr (ptr, <2 x i32>, ...) @llvm.structured.gep.p0.v2i32(ptr elementtype(%S) %src, <2 x i32> <i32 3, i32 4>, i32 0, i32 0)
  ret void
}

define void @out_of_bounds_struct_access(ptr %src) {
entry:
; CHECK: Indexing in a struct should be inbounds
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%S) %src, <1 x i32> <i32 3>, i32 2)
  ret void
}

define void @dynamic_index_struct(ptr %src, i32 %index) {
entry:
; CHECK: Indexing into a struct requires a constant int
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%S) %src, <1 x i32> <i32 3>, i32 %index)
  ret void
}

define void @non_integer_operand(ptr %src) {
entry:
; CHECK: Index operand type must be an integer
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([ 2 x i32 ]) %src, <1 x i32> <i32 5>, float 1.0)
  ret void
}

define void @missing_attribute(ptr %src) {
entry:
; CHECK: Intrinsic first parameter is missing an ElementType attribute
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr %src, <1 x i32> <i32 4>, i32 0)
  ret void
}

define void @amdgpu_stridemark_offset_in_structured_gep(ptr addrspace(9) %src) {
entry:
; CHECK: Reached a non-composite type with more indices to process
  %ptr = call ptr addrspace(9) (ptr addrspace(9), <2 x i32>, ...) @llvm.structured.gep.p9.v2i32(ptr addrspace(9) elementtype([0 x target("amdgpu.stridemark")]) %src, <2 x i32> <i32 4, i32 4>, i32 0, i32 0)
  ret void
}

define void @struct_access_missing_flags(ptr %src) {
entry:
; CHECK: Indexing into a struct requires inbounds and nneg flags
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%S) %src, <1 x i32> zeroinitializer, i32 0)
  ret void
}

define void @flags_wrong_element_count(ptr %src) {
entry:
; CHECK: Flags operand must have one element per index
  %ptr = call ptr (ptr, <2 x i32>, ...) @llvm.structured.gep.p0.v2i32(ptr elementtype([ 2 x i32 ]) %src, <2 x i32> <i32 5, i32 5>, i32 0)
  ret void
}

define void @flags_wrong_element_type(ptr %src) {
entry:
; CHECK: Flags operand type must be a fixed vector of i32
  %ptr = call ptr (ptr, <1 x i64>, ...) @llvm.structured.gep.p0.v1i64(ptr elementtype([ 2 x i32 ]) %src, <1 x i64> <i64 5>, i32 0)
  ret void
}

define void @flags_poison(ptr %src) {
entry:
; CHECK: Flags operand elements must be integer constants
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([ 2 x i32 ]) %src, <1 x i32> poison, i32 0)
  ret void
}

define void @no_index_nonzero_flags(ptr %src) {
entry:
; CHECK: Flags operand must be zero when there are no indices
  %ptr = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(i32) %src, <1 x i32> <i32 1>)
  ret void
}
