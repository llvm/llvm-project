; RUN: not llc -mtriple=amdgpu7.00 < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: error: <unknown>:0:0: in function class_f16 void (ptr addrspace(1), ptr addrspace(1), ptr addrspace(1)): llvm.amdgcn.class only supports f16, f32, and f64

declare i1 @llvm.amdgcn.class.f16(half %a, i32 %b)

define amdgpu_kernel void @class_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load i32, ptr addrspace(1) %b
  %r.val = call i1 @llvm.amdgcn.class.f16(half %a.val, i32 %b.val)
  %r.val.sext = sext i1 %r.val to i32
  store i32 %r.val.sext, ptr addrspace(1) %r
  ret void
}
