;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: UINT_TO_FLT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_kernel void @test(float addrspace(1)* %out, i8 addrspace(1)* %in) {
  %1 = load i8, i8 addrspace(1)* %in
  %2 = uitofp i8 %1 to double
  %3 = fptrunc double %2 to float
  store float %3, float addrspace(1)* %out
  ret void
}
