; RUN: split-file %s %t

;--- test0.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test0.ll | FileCheck %t/test0.ll
; CHECK: intrinsic return type expected void, but got float
; CHECK-NEXT: ptr @llvm.set.rounding

declare float @llvm.set.rounding(i32)

define void @test0() {
entry:
  %t = call float @llvm.set.rounding(i32 0)
  ret void
}

;--- test1.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test1.ll | FileCheck %t/test1.ll
; CHECK: intrinsic return type expected x86_mmx (<1 x i64>), but got x86_amx
; CHECK-NEXT: ptr @llvm.x86.sse.cvtps2pi

declare x86_amx @llvm.x86.sse.cvtps2pi(<4 x float>)

define void @test1(<4 x float> %a) {
entry:
  %t = call x86_amx @llvm.x86.sse.cvtps2pi(<4 x float> %a)
  ret void
}

;--- test2.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test2.ll | FileCheck %t/test2.ll
; CHECK: intrinsic argument 1 type expected x86_mmx (<1 x i64>), but got i32
; CHECK-NEXT: ptr @llvm.x86.sse.cvtpi2ps

declare <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float>, i32)

define void @test1(<4 x float> %a) {
entry:
  %t = call <4 x float> @llvm.x86.sse.cvtpi2ps(<4 x float> %a, i32 0)
  ret void
}

;--- test3.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test3.ll | FileCheck %t/test3.ll
; CHECK: intrinsic return type expected x86_amx, but got float
; CHECK-NEXT: ptr @llvm.x86.tileloadd64.internal

declare float @llvm.x86.tileloadd64.internal(i16, i16, ptr, i64)

define void @test1(ptr %a) {
entry:
  %t = call float @llvm.x86.tileloadd64.internal(i16 0, i16 0, ptr %a, i64 0)
  ret void
}

;--- test4.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test4.ll | FileCheck %t/test4.ll
; CHECK: intrinsic return type expected token, but got i32
; CHECK-NEXT: ptr @llvm.call.preallocated.setup

declare i32 @llvm.call.preallocated.setup(i32)

define void @test1() {
entry:
  %t = call i32 @llvm.call.preallocated.setup(i32 0)
  ret void
}

;--- test5.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test5.ll | FileCheck %t/test5.ll
; CHECK: intrinsic argument 1 type expected metadata, but got i16
; CHECK-NEXT: ptr @llvm.fptrunc.round.f64.f64

declare double @llvm.fptrunc.round.f64.f64(double, i16)

define void @test1() {
entry:
  %t = call double @llvm.fptrunc.round.f64.f64(double 1.0, i16 0)
  ret void
}

;--- test6.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test6.ll | FileCheck %t/test6.ll
; CHECK: intrinsic argument 0 type expected half, but got i16
; CHECK-NEXT: ptr @llvm.nvvm.mul.rn.sat.f16

declare half @llvm.nvvm.mul.rn.sat.f16(i16, half)

define void @test1() {
entry:
  %t = call half @llvm.nvvm.mul.rn.sat.f16(i16 0, half 1.0)
  ret void
}

;--- test7.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test7.ll | FileCheck %t/test7.ll
; CHECK: intrinsic return type expected bfloat, but got half
; CHECK-NEXT: ptr @llvm.arm.neon.vcvtbfp2bf

declare half @llvm.arm.neon.vcvtbfp2bf(float)

define void @test1() {
entry:
  %t = call half @llvm.arm.neon.vcvtbfp2bf(float 0.0)
  ret void
}

;--- test8.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test8.ll | FileCheck %t/test8.ll
; CHECK: intrinsic argument 2 type expected float, but got half
; CHECK-NEXT: ptr @llvm.x86.avx512.vfmadd.f32

declare float @llvm.x86.avx512.vfmadd.f32(float, float, half, i32)

define void @test1() {
entry:
  %t = call float @llvm.x86.avx512.vfmadd.f32(float 0.0, float 0.0, half 0.0, i32 0)
  ret void
}

;--- test9.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test9.ll | FileCheck %t/test9.ll
; CHECK: intrinsic argument 2 type expected double, but got half
; CHECK-NEXT: ptr @llvm.expect.with.probability.i32

declare i32 @llvm.expect.with.probability.i32(i32, i32, half)

define void @test1() {
entry:
  %t = call i32 @llvm.expect.with.probability.i32(i32 0, i32 0, half 0.0)
  ret void
}

;--- test10.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test10.ll | FileCheck %t/test10.ll
; CHECK: intrinsic argument 0 type expected fp128, but got double
; CHECK-NEXT: ptr @llvm.ppc.truncf128.round.to.odd

declare double @llvm.ppc.truncf128.round.to.odd(double)

define void @test1() {
entry:
  %t = call double @llvm.ppc.truncf128.round.to.odd(double 0.0)
  ret void
}

;--- test11.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test11.ll | FileCheck %t/test11.ll
; CHECK: intrinsic argument 0 type expected ppc_fp128, but got double
; CHECK-NEXT: ptr  @llvm.ppc.unpack.longdouble

declare double @llvm.ppc.unpack.longdouble(double, i32)

define void @test1() {
entry:
  %t = call double @llvm.ppc.unpack.longdouble(double 0.0, i32 0)
  ret void
}

;--- test12.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test12.ll | FileCheck %t/test12.ll
; CHECK: intrinsic return type expected i64, but got double
; CHECK-NEXT: ptr @llvm.readcyclecounter

declare double @llvm.readcyclecounter()

define void @test1() {
entry:
  %t = call double @llvm.readcyclecounter()
  ret void
}

;--- test13.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test13.ll | FileCheck %t/test13.ll
; CHECK: intrinsic argument 0 type expected aarch64.svcount, but got i32
; CHECK-NEXT: ptr @llvm.aarch64.sve.pext

declare <4 x i32> @llvm.aarch64.sve.pext(i32, i32)

define void @test1() {
entry:
  %t = call <4 x i32> @llvm.aarch64.sve.pext(i32 0, i32 0)
  ret void
}

;--- test14.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test14.ll | FileCheck %t/test14.ll
; CHECK: intrinsic return type expected vector with 16 elements, but got i32
; CHECK-NEXT: ptr @llvm.aarch64.neon.pmull64

; Correct return type is <16 x i8>
declare i32 @llvm.aarch64.neon.pmull64(i64, i64)

define void @test1() {
entry:
  %t = call i32 @llvm.aarch64.neon.pmull64(i64 0, i64 0)
  ret void
}

;--- test15.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test15.ll | FileCheck %t/test15.ll
; CHECK: intrinsic return vector element type expected i8, but got i32
; CHECK-NEXT: ptr @llvm.aarch64.neon.pmull64

declare <16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)

define void @test1() {
entry:
  %t = call <16 x i32> @llvm.aarch64.neon.pmull64(i64 0, i64 0)
  ret void
}

;--- test16.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test16.ll | FileCheck %t/test16.ll
; CHECK: intrinsic return type expected vector with 16 elements, but got <vscale x 16 x i32>
; CHECK-NEXT: ptr @llvm.aarch64.neon.pmull64

declare <vscale x 16 x i32> @llvm.aarch64.neon.pmull64(i64, i64)

define void @test1() {
entry:
  %t = call <vscale x 16 x i32> @llvm.aarch64.neon.pmull64(i64 0, i64 0)
  ret void
}

;--- test17.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test17.ll | FileCheck %t/test17.ll
; CHECK: intrinsic argument 1 type expected ptr, but got i32
; CHECK-NEXT: ptr @llvm.gcroot

declare void @llvm.gcroot(ptr, i32)

define void @test1() {
entry:
  call void @llvm.gcroot(ptr null, i32 0)
  ret void
}

;--- test18.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test18.ll | FileCheck %t/test18.ll
; CHECK: intrinsic argument 0 type expected ptr addrspace(1), but got ptr
; CHECK-NEXT: ptr @llvm.nvvm.applypriority.global.L2.evict.normal

declare void @llvm.nvvm.applypriority.global.L2.evict.normal(ptr, i64)

define void @test1() {
entry:
  call void @llvm.nvvm.applypriority.global.L2.evict.normal(ptr null, i64 0)
  ret void
}

;--- test19.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test19.ll | FileCheck %t/test19.ll
; CHECK: intrinsic return type expected literal non-packed struct with 2 elements, but got void
; CHECK-NEXT: ptr @llvm.nvvm.elect.sync

; Expected return type is { i32, i1 }.
declare void @llvm.nvvm.elect.sync(i32)

define void @test1() {
entry:
  call void @llvm.nvvm.elect.sync(i32 0)
  ret void
}

;--- test20.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test20.ll | FileCheck %t/test20.ll
; CHECK: intrinsic return struct element 1 type expected i1, but got i2
; CHECK-NEXT: ptr @llvm.nvvm.elect.sync

; Expected return type is { i32, i1 }.
declare { i32, i2 } @llvm.nvvm.elect.sync(i32)

define void @test1() {
entry:
  call { i32, i2 } @llvm.nvvm.elect.sync(i32 0)
  ret void
}

;--- test21.ll
; RUN: not opt -S -passes=verify 2>&1 < %t/test21.ll | FileCheck %t/test21.ll
; CHECK: intrinsic return type expected literal non-packed struct with 2 elements, but got { i32, i1, i1 }
; CHECK-NEXT: ptr @llvm.nvvm.elect.sync

; Expected return type is { i32, i1 }.
declare { i32, i1, i1 } @llvm.nvvm.elect.sync(i32)

define void @test1() {
entry:
  call { i32, i1, i1 } @llvm.nvvm.elect.sync(i32 0)
  ret void
}
