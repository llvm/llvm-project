;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK-NOT: SETE_INT
;CHECK: CNDE_INT {{\*?}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, 1, literal.x,
;CHECK-NEXT: 2
define amdgpu_kernel void @test(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %1 = load i32, i32 addrspace(1)* %in
  %2 = icmp eq i32 %1, 0
  %3 = select i1 %2, i32 1, i32 2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}
