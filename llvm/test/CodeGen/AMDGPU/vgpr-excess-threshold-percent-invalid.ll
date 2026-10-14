; Test that invalid values for -amdgpu-vgpr-threshold-percent are rejected.

; RUN: not llc -mtriple=amdgpu9.42 -amdgpu-vgpr-threshold-percent=-20 -o /dev/null %s 2>&1 | FileCheck -check-prefix=NEGATIVE %s
; RUN: not llc -mtriple=amdgpu9.42 -amdgpu-vgpr-threshold-percent=140 -o /dev/null %s 2>&1 | FileCheck -check-prefix=OVER100 %s

; NEGATIVE: for the --amdgpu-vgpr-threshold-percent option: '-20' value invalid for uint argument!
; OVER100: for the --amdgpu-vgpr-threshold-percent option: '140' value must be in the range [0, 100]!

define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load float, ptr addrspace(1) %in
  store float %val, ptr addrspace(1) %out
  ret void
}
