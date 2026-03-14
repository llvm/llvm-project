;RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s

;CHECK-NOT: SETE
;CHECK: CNDE {{\*?}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1.0, literal.x,
;CHECK: 1073741824
define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %1 = load float, ptr addrspace(1) %in
  %2 = fcmp oeq float %1, 0.0
  %3 = select i1 %2, float 1.0, float 2.0
  store float %3, ptr addrspace(1) %out
  ret void
}
