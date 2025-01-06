; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_32 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_32 | %ptxas-verify %}


declare i8 @llvm.nvvm.ldu.global.i.i8.p1(ptr addrspace(1) %ptr, i32 %align)
declare i16 @llvm.nvvm.ldu.global.i.i16.p1(ptr addrspace(1) %ptr, i32 %align)
declare i32 @llvm.nvvm.ldu.global.i.i32.p1(ptr addrspace(1) %ptr, i32 %align)
declare i64 @llvm.nvvm.ldu.global.i.i64.p1(ptr addrspace(1) %ptr, i32 %align)
declare ptr @llvm.nvvm.ldu.global.p.p1(ptr addrspace(1) %ptr, i32 %align)
declare float @llvm.nvvm.ldu.global.f.f32.p1(ptr addrspace(1) %ptr, i32 %align)
declare double @llvm.nvvm.ldu.global.f.f64.p1(ptr addrspace(1) %ptr, i32 %align)
declare half @llvm.nvvm.ldu.global.f.f16.p1(ptr addrspace(1) %ptr, i32 %align)
declare <2 x half> @llvm.nvvm.ldu.global.f.v2f16.p1(ptr addrspace(1) %ptr, i32 %align)

declare i8 @llvm.nvvm.ldg.global.i.i8.p1(ptr addrspace(1) %ptr, i32 %align)
declare i16 @llvm.nvvm.ldg.global.i.i16.p1(ptr addrspace(1) %ptr, i32 %align)
declare i32 @llvm.nvvm.ldg.global.i.i32.p1(ptr addrspace(1) %ptr, i32 %align)
declare i64 @llvm.nvvm.ldg.global.i.i64.p1(ptr addrspace(1) %ptr, i32 %align)
declare ptr @llvm.nvvm.ldg.global.p.p1(ptr addrspace(1) %ptr, i32 %align)
declare float @llvm.nvvm.ldg.global.f.f32.p1(ptr addrspace(1) %ptr, i32 %align)
declare double @llvm.nvvm.ldg.global.f.f64.p1(ptr addrspace(1) %ptr, i32 %align)
declare half @llvm.nvvm.ldg.global.f.f16.p1(ptr addrspace(1) %ptr, i32 %align)
declare <2 x half> @llvm.nvvm.ldg.global.f.v2f16.p1(ptr addrspace(1) %ptr, i32 %align)

; CHECK-LABEL: test_ldu_i8
define i8 @test_ldu_i8(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.u8
  %val = tail call i8 @llvm.nvvm.ldu.global.i.i8.p1(ptr addrspace(1) %ptr, i32 4)
  ret i8 %val
}

; CHECK-LABEL: test_ldu_i16
define i16 @test_ldu_i16(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.u16
  %val = tail call i16 @llvm.nvvm.ldu.global.i.i16.p1(ptr addrspace(1) %ptr, i32 2)
  ret i16 %val
}

; CHECK-LABEL: test_ldu_i32
define i32 @test_ldu_i32(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.u32
  %val = tail call i32 @llvm.nvvm.ldu.global.i.i32.p1(ptr addrspace(1) %ptr, i32 4)
  ret i32 %val
}

; CHECK-LABEL: test_ldu_i64
define i64 @test_ldu_i64(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.u64
  %val = tail call i64 @llvm.nvvm.ldu.global.i.i64.p1(ptr addrspace(1) %ptr, i32 8)
  ret i64 %val
}

; CHECK-LABEL: test_ldu_p
define ptr @test_ldu_p(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.u64
  %val = tail call ptr @llvm.nvvm.ldu.global.p.p1(ptr addrspace(1) %ptr, i32 8)
  ret ptr %val
}


; CHECK-LABEL: test_ldu_f32
define float @test_ldu_f32(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.f32
  %val = tail call float @llvm.nvvm.ldu.global.f.f32.p1(ptr addrspace(1) %ptr, i32 4)
  ret float %val
}

; CHECK-LABEL: test_ldu_f64
define double @test_ldu_f64(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.f64
  %val = tail call double @llvm.nvvm.ldu.global.f.f64.p1(ptr addrspace(1) %ptr, i32 8)
  ret double %val
}

; CHECK-LABEL: test_ldu_f16
define half @test_ldu_f16(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.u16
  %val = tail call half @llvm.nvvm.ldu.global.f.f16.p1(ptr addrspace(1) %ptr, i32 2)
  ret half %val
}

; CHECK-LABEL: test_ldu_v2f16
define <2 x half> @test_ldu_v2f16(ptr addrspace(1) %ptr) {
  ; CHECK: ldu.global.u32
  %val = tail call <2 x half> @llvm.nvvm.ldu.global.f.v2f16.p1(ptr addrspace(1) %ptr, i32 4)
  ret <2 x half> %val
}

; CHECK-LABEL: test_ldg_i8
define i8 @test_ldg_i8(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.u8
  %val = tail call i8 @llvm.nvvm.ldg.global.i.i8.p1(ptr addrspace(1) %ptr, i32 4)
  ret i8 %val
}

; CHECK-LABEL: test_ldg_i16
define i16 @test_ldg_i16(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.u16
  %val = tail call i16 @llvm.nvvm.ldg.global.i.i16.p1(ptr addrspace(1) %ptr, i32 2)
  ret i16 %val
}

; CHECK-LABEL: test_ldg_i32
define i32 @test_ldg_i32(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.u32
  %val = tail call i32 @llvm.nvvm.ldg.global.i.i32.p1(ptr addrspace(1) %ptr, i32 4)
  ret i32 %val
}

; CHECK-LABEL: test_ldg_i64
define i64 @test_ldg_i64(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.u64
  %val = tail call i64 @llvm.nvvm.ldg.global.i.i64.p1(ptr addrspace(1) %ptr, i32 8)
  ret i64 %val
}

; CHECK-LABEL: test_ldg_p
define ptr @test_ldg_p(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.u64
  %val = tail call ptr @llvm.nvvm.ldg.global.p.p1(ptr addrspace(1) %ptr, i32 8)
  ret ptr %val
}

; CHECK-LABEL: test_ldg_f32
define float @test_ldg_f32(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.f32
  %val = tail call float @llvm.nvvm.ldg.global.f.f32.p1(ptr addrspace(1) %ptr, i32 4)
  ret float %val
}

; CHECK-LABEL: test_ldg_f64
define double @test_ldg_f64(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.f64
  %val = tail call double @llvm.nvvm.ldg.global.f.f64.p1(ptr addrspace(1) %ptr, i32 8)
  ret double %val
}

; CHECK-LABEL: test_ldg_f16
define half @test_ldg_f16(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.u16
  %val = tail call half @llvm.nvvm.ldg.global.f.f16.p1(ptr addrspace(1) %ptr, i32 2)
  ret half %val
}

; CHECK-LABEL: test_ldg_v2f16
define <2 x half> @test_ldg_v2f16(ptr addrspace(1) %ptr) {
  ; CHECK: ld.global.nc.u32
  %val = tail call <2 x half> @llvm.nvvm.ldg.global.f.v2f16.p1(ptr addrspace(1) %ptr, i32 4)
  ret <2 x half> %val
}
