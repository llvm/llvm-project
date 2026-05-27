; RUN: llvm-as -disable-output %s

define void @dynamic_index_then_gep_offset(ptr addrspace(9) %p, i32 %index,
                                           i32 %offset) {
entry:
  %q1 = call ptr addrspace(9) (ptr addrspace(9), <1 x i32>, ...) @llvm.structured.gep.p9.v1i32(ptr addrspace(9) elementtype([0 x target("amdgpu.stridemark")]) %p, <1 x i32> <i32 4>, i32 %index)
  %q = getelementptr i8, ptr addrspace(9) %q1, i32 %offset
  ret void
}

define void @known_bounds_and_stride(ptr addrspace(9) %p, i32 %index,
                                     i32 %offset) {
entry:
  %q1 = call ptr addrspace(9) (ptr addrspace(9), <1 x i32>, ...) @llvm.structured.gep.p9.v1i32(ptr addrspace(9) elementtype([256 x target("amdgpu.stridemark", 16)]) %p, <1 x i32> <i32 5>, i32 %index)
  %q = getelementptr i8, ptr addrspace(9) %q1, i32 %offset
  ret void
}
