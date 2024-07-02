; RUN: not --crash opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s 2>&1 | FileCheck %s
; RUN: not --crash opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s 2>&1 | FileCheck %s

@var1 = addrspace(3) global i32 undef, !absolute_symbol !0
@var2 = addrspace(3) global i32 undef

; CHECK: Module cannot mix absolute and non-absolute LDS GVs
define amdgpu_kernel void @kern() {
  %val0 = load i32, ptr addrspace(3) @var1
  %val1 = add i32 %val0, 4
  store i32 %val1, ptr addrspace(3) @var2
  ret void
}

!0 = !{i32 0, i32 1}
