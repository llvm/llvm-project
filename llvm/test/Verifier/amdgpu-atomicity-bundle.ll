; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @f()

; CHECK: "amdgpu.atomicity" operand bundle is only valid on AMDGPU buffer memory intrinsics
define void @not_a_buffer_intrinsic() {
  call void @f() [ "amdgpu.atomicity"(metadata !"release", metadata !"agent") ]
  ret void
}

; CHECK: "amdgpu.atomicity" operand bundle is only valid on AMDGPU buffer memory intrinsics
define void @wrong_intrinsic(ptr addrspace(8) %rsrc) {
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) null, i32 4, i32 0, i32 0, i32 0, i32 0) [ "amdgpu.atomicity"(metadata !"release", metadata !"agent") ]
  ret void
}

; CHECK: Multiple "amdgpu.atomicity" operand bundles
define void @multiple(i32 %val, ptr addrspace(8) %rsrc) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %val, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0) [ "amdgpu.atomicity"(metadata !"release", metadata !"agent"), "amdgpu.atomicity"(metadata !"release", metadata !"agent") ]
  ret void
}

; CHECK: Expected exactly two "amdgpu.atomicity" bundle operands
define void @wrong_operand_count(i32 %val, ptr addrspace(8) %rsrc) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %val, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0) [ "amdgpu.atomicity"(metadata !"release") ]
  ret void
}

; CHECK: "amdgpu.atomicity" ordering operand must be a metadata string naming an atomic ordering
define void @not_an_ordering(i32 %val, ptr addrspace(8) %rsrc) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %val, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0) [ "amdgpu.atomicity"(metadata !"consume", metadata !"agent") ]
  ret void
}

; CHECK: "amdgpu.atomicity" ordering operand must be a metadata string naming an atomic ordering
define void @ordering_not_a_string(i32 %val, ptr addrspace(8) %rsrc) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %val, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0) [ "amdgpu.atomicity"(i32 0, metadata !"agent") ]
  ret void
}

; CHECK: "amdgpu.atomicity" syncscope operand must be a metadata string
define void @scope_not_a_string(i32 %val, ptr addrspace(8) %rsrc) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %val, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0) [ "amdgpu.atomicity"(metadata !"release", i32 0) ]
  ret void
}
