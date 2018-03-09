; RUN: opt -mtriple=r600---amdgiz -O3 -aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; CHECK: NoAlias: i8 addrspace(5)* %p, i8 addrspace(7)* %p1

define amdgpu_kernel void @test(i8 addrspace(5)* %p, i8 addrspace(7)* %p1) {
  ret void
}
