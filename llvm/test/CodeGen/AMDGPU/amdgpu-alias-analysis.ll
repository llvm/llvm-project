; RUN: opt -mtriple=amdgcn-- -data-layout=A5 -aa-eval -amdgpu-aa -amdgpu-aa-wrapper -disable-basic-aa  -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=r600-- -data-layout=A5 -aa-eval -amdgpu-aa -amdgpu-aa-wrapper -disable-basic-aa  -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=amdgcn-- -data-layout=A5 -passes=aa-eval -aa-pipeline=amdgpu-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=r600-- -data-layout=A5 -passes=aa-eval -aa-pipeline=amdgpu-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

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

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8* %p1
define void @test_region_vs_flat(i8 addrspace(2)* %p, i8 addrspace(0)* %p1) {
  ret void
}

; CHECK: NoAlias:  i8 addrspace(1)* %p1, i8 addrspace(2)* %p
define void @test_region_vs_global(i8 addrspace(2)* %p, i8 addrspace(1)* %p1) {
  ret void
}

; CHECK: MayAlias: i8 addrspace(2)* %p, i8 addrspace(2)* %p1
define void @test_region(i8 addrspace(2)* %p, i8 addrspace(2)* %p1) {
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(3)* %p1
define void @test_region_vs_group(i8 addrspace(2)* %p, i8 addrspace(3)* %p1) {
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(4)* %p1
define void @test_region_vs_constant(i8 addrspace(2)* %p, i8 addrspace(4)* %p1) {
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(5)* %p1
define void @test_region_vs_private(i8 addrspace(2)* %p, i8 addrspace(5)* %p1) {
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(6)* %p1
define void @test_region_vs_const32(i8 addrspace(2)* %p, i8 addrspace(6)* %p1) {
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

@cst = internal addrspace(4) global i8* undef, align 4

; CHECK-LABEL: Function: test_8_0
; CHECK: NoAlias:   i8 addrspace(3)* %p, i8* %p1
; CHECK: NoAlias:   i8 addrspace(3)* %p, i8* addrspace(4)* @cst
; CHECK: MayAlias:  i8* %p1, i8* addrspace(4)* @cst
define void @test_8_0(i8 addrspace(3)* %p) {
  %p1 = load i8*, i8* addrspace(4)* @cst
  ret void
}

; CHECK-LABEL: Function: test_8_1
; CHECK: NoAlias:   i8 addrspace(5)* %p, i8* %p1
; CHECK: NoAlias:   i8 addrspace(5)* %p, i8* addrspace(4)* @cst
; CHECK: MayAlias:  i8* %p1, i8* addrspace(4)* @cst
define void @test_8_1(i8 addrspace(5)* %p) {
  %p1 = load i8*, i8* addrspace(4)* @cst
  ret void
}

; CHECK-LABEL: Function: test_8_2
; CHECK: NoAlias:   i8 addrspace(5)* %p1, i8* %p
define amdgpu_kernel void @test_8_2(i8* %p) {
  %p1 = alloca i8, align 1, addrspace(5)
  ret void
}

; CHECK-LABEL: Function: test_8_3
; CHECK: MayAlias:  i8 addrspace(5)* %p1, i8* %p
; TODO: So far, %p1 may still alias to %p. As it's not captured at all, it
; should be NoAlias.
define void @test_8_3(i8* %p) {
  %p1 = alloca i8, align 1, addrspace(5)
  ret void
}

@shm = internal addrspace(3) global i8 undef, align 4

; CHECK-LABEL: Function: test_8_4
; CHECK: NoAlias:   i8 addrspace(3)* %p1, i8* %p
; CHECK: NoAlias:   i8 addrspace(3)* @shm, i8* %p
; CHECK: MayAlias:  i8 addrspace(3)* %p1, i8 addrspace(3)* @shm
define amdgpu_kernel void @test_8_4(i8* %p) {
  %p1 = getelementptr i8, i8 addrspace(3)* @shm, i32 0
  ret void
}

; CHECK-LABEL: Function: test_8_5
; CHECK: MayAlias:  i8 addrspace(3)* %p1, i8* %p
; CHECK: MayAlias:  i8 addrspace(3)* @shm, i8* %p
; CHECK: MayAlias:  i8 addrspace(3)* %p1, i8 addrspace(3)* @shm
; TODO: So far, %p1 may still alias to %p. As it's not captured at all, it
; should be NoAlias.
define void @test_8_5(i8* %p) {
  %p1 = getelementptr i8, i8 addrspace(3)* @shm, i32 0
  ret void
}
