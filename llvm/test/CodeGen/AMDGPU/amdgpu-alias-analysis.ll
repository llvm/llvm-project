; RUN: opt -mtriple=amdgcn-- -O3 -aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=r600-- -O3 -aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; CHECK: NoAlias:      i8 addrspace(1)* %p1, i8 addrspace(5)* %p

define void @test(i8 addrspace(5)* %p, i8 addrspace(1)* %p1) {
  ret void
}

; CHECK: MayAlias:      i8 addrspace(1)* %p1, i8 addrspace(4)* %p

define void @test_constant_vs_global(i8 addrspace(4)* %p, i8 addrspace(1)* %p1) {
  ret void
}

; CHECK: MayAlias:      i8 addrspace(1)* %p, i8 addrspace(4)* %p1

define void @test_global_vs_constant(i8 addrspace(1)* %p, i8 addrspace(4)* %p1) {
  ret void
}

; CHECK: MayAlias:      i8 addrspace(1)* %p1, i8 addrspace(6)* %p

define void @test_constant_32bit_vs_global(i8 addrspace(6)* %p, i8 addrspace(1)* %p1) {
  ret void
}

; CHECK: MayAlias:      i8 addrspace(4)* %p1, i8 addrspace(6)* %p

define void @test_constant_32bit_vs_constant(i8 addrspace(6)* %p, i8 addrspace(4)* %p1) {
  ret void
}

