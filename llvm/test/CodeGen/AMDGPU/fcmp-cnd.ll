;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;Not checking arguments 2 and 3 to CNDE, because they may change between
;registers and literal.x depending on what the optimizer does.
;CHECK: CNDE  T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_kernel void @test(i32 addrspace(1)* %out, float addrspace(1)* %in) {
entry:
  %0 = load float, float addrspace(1)* %in
  %cmp = fcmp oeq float %0, 0.000000e+00
  %value = select i1 %cmp, i32 2, i32 3 
  store i32 %value, i32 addrspace(1)* %out
  ret void
}
