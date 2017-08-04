; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=HSA %s

@internal_group = internal addrspace(3) global i32 undef
@external_group = addrspace(3) global i32 undef

define amdgpu_kernel void @test() {
entry:
  store i32 0, i32 addrspace(3)* @internal_group
  store i32 0, i32 addrspace(3)* @external_group
  ret void
}

; HSA-NOT: internal_group:
; HSA-NOT: external_group:
