; RUN: opt -mtriple=amdgcn-- -aa-eval -amdgpu-aa -amdgpu-aa-wrapper -disable-basicaa  -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=r600-- -aa-eval -amdgpu-aa -amdgpu-aa-wrapper -disable-basicaa  -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

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

; CHECK: MayAlias:	i8 addrspace(999)* %p0, i8* %p
define void @test_0_999(i8 addrspace(0)* %p, i8 addrspace(999)* %p0) {
  ret void
}

; CHECK: MayAlias:	i8 addrspace(999)* %p, i8* %p1
define void @test_999_0(i8 addrspace(999)* %p, i8 addrspace(0)* %p1) {
  ret void
}

; CHECK: MayAlias:	i8 addrspace(1)* %p, i8 addrspace(999)* %p1
define void @test_1_999(i8 addrspace(1)* %p, i8 addrspace(999)* %p1) {
  ret void
}

; CHECK: MayAlias:	i8 addrspace(1)* %p1, i8 addrspace(999)* %p
define void @test_999_1(i8 addrspace(999)* %p, i8 addrspace(1)* %p1) {
  ret void
}

; CHECK: MayAlias:  i8 addrspace(7)* %p, i8* %p1
define void @test_7_0(i8 addrspace(7)* %p, i8 addrspace(0)* %p1) {
    ret void
}

; CHECK: MayAlias:  i8 addrspace(1)* %p1, i8 addrspace(7)* %p
define void @test_7_1(i8 addrspace(7)* %p, i8 addrspace(1)* %p1) {
    ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p1, i8 addrspace(7)* %p
define void @test_7_2(i8 addrspace(7)* %p, i8 addrspace(2)* %p1) {
    ret void
}

; CHECK: NoAlias:  i8 addrspace(3)* %p1, i8 addrspace(7)* %p
define void @test_7_3(i8 addrspace(7)* %p, i8 addrspace(3)* %p1) {
    ret void
}

; CHECK: MayAlias:  i8 addrspace(4)* %p1, i8 addrspace(7)* %p
define void @test_7_4(i8 addrspace(7)* %p, i8 addrspace(4)* %p1) {
    ret void
}

; CHECK: NoAlias:  i8 addrspace(5)* %p1, i8 addrspace(7)* %p
define void @test_7_5(i8 addrspace(7)* %p, i8 addrspace(5)* %p1) {
    ret void
}

; CHECK: MayAlias:  i8 addrspace(6)* %p1, i8 addrspace(7)* %p
define void @test_7_6(i8 addrspace(7)* %p, i8 addrspace(6)* %p1) {
    ret void
}

; CHECK: MayAlias:  i8 addrspace(7)* %p, i8 addrspace(7)* %p1
define void @test_7_7(i8 addrspace(7)* %p, i8 addrspace(7)* %p1) {
    ret void
}
