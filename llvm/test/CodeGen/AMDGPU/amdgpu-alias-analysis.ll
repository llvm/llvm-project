; RUN: opt -mtriple=amdgcn-- -data-layout=A5 -passes=aa-eval -aa-pipeline=amdgpu-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=r600-- -data-layout=A5 -passes=aa-eval -aa-pipeline=amdgpu-aa -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; CHECK-LABEL: Function: test
; CHECK: NoAlias:      i8 addrspace(5)* %p, i8 addrspace(1)* %p1

define void @test(ptr addrspace(5) %p, ptr addrspace(1) %p1) {
  load i8, ptr addrspace(5) %p
  load i8, ptr addrspace(1) %p1
  ret void
}

; CHECK-LABEL: Function: test_constant_vs_global
; CHECK: MayAlias:      i8 addrspace(4)* %p, i8 addrspace(1)* %p1

define void @test_constant_vs_global(ptr addrspace(4) %p, ptr addrspace(1) %p1) {
  load i8, ptr addrspace(4) %p
  load i8, ptr addrspace(1) %p1
  ret void
}

; CHECK: MayAlias:      i8 addrspace(1)* %p, i8 addrspace(4)* %p1

define void @test_global_vs_constant(ptr addrspace(1) %p, ptr addrspace(4) %p1) {
  load i8, ptr addrspace(1) %p
  load i8, ptr addrspace(4) %p1
  ret void
}

; CHECK: MayAlias:      i8 addrspace(6)* %p, i8 addrspace(1)* %p1

define void @test_constant_32bit_vs_global(ptr addrspace(6) %p, ptr addrspace(1) %p1) {
  load i8, ptr addrspace(6) %p
  load i8, ptr addrspace(1) %p1
  ret void
}

; CHECK: MayAlias:      i8 addrspace(6)* %p, i8 addrspace(4)* %p1

define void @test_constant_32bit_vs_constant(ptr addrspace(6) %p, ptr addrspace(4) %p1) {
  load i8, ptr addrspace(6) %p
  load i8, ptr addrspace(4) %p1
  ret void
}

; CHECK: MayAlias:	i8* %p, i8 addrspace(999)* %p0
define void @test_0_999(ptr addrspace(0) %p, ptr addrspace(999) %p0) {
  load i8, ptr addrspace(0) %p
  load i8, ptr addrspace(999) %p0
  ret void
}

; CHECK: MayAlias:	i8 addrspace(999)* %p, i8* %p1
define void @test_999_0(ptr addrspace(999) %p, ptr addrspace(0) %p1) {
  load i8, ptr addrspace(999) %p
  load i8, ptr addrspace(0) %p1
  ret void
}

; CHECK: MayAlias:	i8 addrspace(1)* %p, i8 addrspace(999)* %p1
define void @test_1_999(ptr addrspace(1) %p, ptr addrspace(999) %p1) {
  load i8, ptr addrspace(1) %p
  load i8, ptr addrspace(999) %p1
  ret void
}

; CHECK: MayAlias:	i8 addrspace(999)* %p, i8 addrspace(1)* %p1
define void @test_999_1(ptr addrspace(999) %p, ptr addrspace(1) %p1) {
  load i8, ptr addrspace(999) %p
  load i8, ptr addrspace(1) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8* %p1
define void @test_region_vs_flat(ptr addrspace(2) %p, ptr addrspace(0) %p1) {
  load i8, ptr addrspace(2) %p
  load i8, ptr addrspace(0) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(1)* %p1
define void @test_region_vs_global(ptr addrspace(2) %p, ptr addrspace(1) %p1) {
  load i8, ptr addrspace(2) %p
  load i8, ptr addrspace(1) %p1
  ret void
}

; CHECK: MayAlias: i8 addrspace(2)* %p, i8 addrspace(2)* %p1
define void @test_region(ptr addrspace(2) %p, ptr addrspace(2) %p1) {
  load i8, ptr addrspace(2) %p
  load i8, ptr addrspace(2) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(3)* %p1
define void @test_region_vs_group(ptr addrspace(2) %p, ptr addrspace(3) %p1) {
  load i8, ptr addrspace(2) %p
  load i8, ptr addrspace(3) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(4)* %p1
define void @test_region_vs_constant(ptr addrspace(2) %p, ptr addrspace(4) %p1) {
  load i8, ptr addrspace(2) %p
  load i8, ptr addrspace(4) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(5)* %p1
define void @test_region_vs_private(ptr addrspace(2) %p, ptr addrspace(5) %p1) {
  load i8, ptr addrspace(2) %p
  load i8, ptr addrspace(5) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(2)* %p, i8 addrspace(6)* %p1
define void @test_region_vs_const32(ptr addrspace(2) %p, ptr addrspace(6) %p1) {
  load i8, ptr addrspace(2) %p
  load i8, ptr addrspace(6) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(7)* %p, i8* %p1
define void @test_7_0(ptr addrspace(7) %p, ptr addrspace(0) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(0) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(7)* %p, i8 addrspace(1)* %p1
define void @test_7_1(ptr addrspace(7) %p, ptr addrspace(1) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(1) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(7)* %p, i8 addrspace(2)* %p1
define void @test_7_2(ptr addrspace(7) %p, ptr addrspace(2) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(2) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(7)* %p, i8 addrspace(3)* %p1
define void @test_7_3(ptr addrspace(7) %p, ptr addrspace(3) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(3) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(7)* %p, i8 addrspace(4)* %p1
define void @test_7_4(ptr addrspace(7) %p, ptr addrspace(4) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(4) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(7)* %p, i8 addrspace(5)* %p1
define void @test_7_5(ptr addrspace(7) %p, ptr addrspace(5) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(5) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(7)* %p, i8 addrspace(6)* %p1
define void @test_7_6(ptr addrspace(7) %p, ptr addrspace(6) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(6) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(7)* %p, i8 addrspace(7)* %p1
define void @test_7_7(ptr addrspace(7) %p, ptr addrspace(7) %p1) {
  load i8, ptr addrspace(7) %p
  load i8, ptr addrspace(7) %p1
  ret void
}

@cst = internal addrspace(4) global ptr undef, align 4

; CHECK-LABEL: Function: test_8_0
; CHECK-DAG: NoAlias:   i8 addrspace(3)* %p, i8* %p1
; CHECK-DAG: NoAlias:   i8 addrspace(3)* %p, ptr addrspace(4)* @cst
; CHECK-DAG: MayAlias:  i8* %p1, ptr addrspace(4)* @cst
define void @test_8_0(ptr addrspace(3) %p) {
  %p1 = load ptr, ptr addrspace(4) @cst
  load i8, ptr addrspace(3) %p
  load i8, ptr %p1
  ret void
}

; CHECK-LABEL: Function: test_8_1
; CHECK-DAG: NoAlias:   i8 addrspace(5)* %p, i8* %p1
; CHECK-DAG: NoAlias:   i8 addrspace(5)* %p, ptr addrspace(4)* @cst
; CHECK-DAG: MayAlias:  i8* %p1, ptr addrspace(4)* @cst
define void @test_8_1(ptr addrspace(5) %p) {
  %p1 = load ptr, ptr addrspace(4) @cst
  load i8, ptr addrspace(5) %p
  load i8, ptr %p1
  ret void
}

; CHECK-LABEL: Function: test_8_2
; CHECK: NoAlias:   i8* %p, i8 addrspace(5)* %p1
define amdgpu_kernel void @test_8_2(ptr %p) {
  %p1 = alloca i8, align 1, addrspace(5)
  load i8, ptr %p
  load i8, ptr addrspace(5) %p1
  ret void
}

; CHECK-LABEL: Function: test_8_3
; CHECK: MayAlias:  i8* %p, i8 addrspace(5)* %p1
; TODO: So far, %p1 may still alias to %p. As it's not captured at all, it
; should be NoAlias.
define void @test_8_3(ptr %p) {
  %p1 = alloca i8, align 1, addrspace(5)
  load i8, ptr %p
  load i8, ptr addrspace(5) %p1
  ret void
}

@shm = internal addrspace(3) global [2 x i8] undef, align 4

; CHECK-LABEL: Function: test_8_4
; CHECK: NoAlias:   i8* %p, i8 addrspace(3)* %p1
; CHECK: NoAlias:   i8* %p, i8 addrspace(3)* @shm
; CHECK: MayAlias:  i8 addrspace(3)* %p1, i8 addrspace(3)* @shm
define amdgpu_kernel void @test_8_4(ptr %p) {
  %p1 = getelementptr [2 x i8], ptr addrspace(3) @shm, i32 0, i32 1
  load i8, ptr %p
  load i8, ptr addrspace(3) %p1
  load i8, ptr addrspace(3) @shm
  ret void
}

; CHECK-LABEL: Function: test_8_5
; CHECK: MayAlias:  i8* %p, i8 addrspace(3)* %p1
; CHECK: MayAlias:  i8* %p, i8 addrspace(3)* @shm
; CHECK: MayAlias:  i8 addrspace(3)* %p1, i8 addrspace(3)* @shm

; TODO: So far, @shm may still alias to %p. As it's not captured at all, it
; should be NoAlias.
define void @test_8_5(ptr %p) {
  %p1 = getelementptr [2 x i8], ptr addrspace(3) @shm, i32 0, i32 1
  load i8, ptr %p
  load i8, ptr addrspace(3) %p1
  load i8, ptr addrspace(3) @shm
  ret void
}

; CHECK: MayAlias:  i8 addrspace(9)* %p, i8* %p1
define void @test_9_0(ptr addrspace(9) %p, ptr addrspace(0) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(0) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(9)* %p, i8 addrspace(1)* %p1
define void @test_9_1(ptr addrspace(9) %p, ptr addrspace(1) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(1) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(9)* %p, i8 addrspace(2)* %p1
define void @test_9_2(ptr addrspace(9) %p, ptr addrspace(2) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(2) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(9)* %p, i8 addrspace(3)* %p1
define void @test_9_3(ptr addrspace(9) %p, ptr addrspace(3) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(3) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(9)* %p, i8 addrspace(4)* %p1
define void @test_9_4(ptr addrspace(9) %p, ptr addrspace(4) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(4) %p1
  ret void
}

; CHECK: NoAlias:  i8 addrspace(9)* %p, i8 addrspace(5)* %p1
define void @test_9_5(ptr addrspace(9) %p, ptr addrspace(5) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(5) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(9)* %p, i8 addrspace(6)* %p1
define void @test_9_6(ptr addrspace(9) %p, ptr addrspace(6) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(6) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(9)* %p, i8 addrspace(7)* %p1
define void @test_9_7(ptr addrspace(9) %p, ptr addrspace(7) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(7) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(9)* %p, i8 addrspace(8)* %p1
define void @test_9_8(ptr addrspace(9) %p, ptr addrspace(8) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(8) %p1
  ret void
}

; CHECK: MayAlias:  i8 addrspace(9)* %p, i8 addrspace(9)* %p1
define void @test_9_9(ptr addrspace(9) %p, ptr addrspace(9) %p1) {
  load i8, ptr addrspace(9) %p
  load i8, ptr addrspace(9) %p1
  ret void
}
