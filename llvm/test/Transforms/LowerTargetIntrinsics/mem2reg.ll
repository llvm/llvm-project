; RUN: opt -S -passes=lower-target-intrinsics -mtriple=x86_64-unknown-linux -mcpu=haswell < %s | FileCheck %s --check-prefix=HASWELL
; RUN: opt -S -passes=lower-target-intrinsics -mtriple=x86_64-unknown-linux -mcpu=x86-64 < %s | FileCheck %s --check-prefix=GENERIC

; This test simulates -O0 codegen where Clang emits alloca/store/load instead
; of SSA values. The pass must promote these to SSA before resolution.

declare void @avx2_path()
declare void @fallback_path()

define void @test_o0_alloca_i1() {
; HASWELL-LABEL: @test_o0_alloca_i1(
; HASWELL-NOT:   @fallback_path
; HASWELL:       call void @avx2_path()
; HASWELL:       ret void
;
; GENERIC-LABEL: @test_o0_alloca_i1(
; GENERIC-NOT:   @avx2_path
; GENERIC:       call void @fallback_path()
; GENERIC:       ret void
entry:
  %has_avx2.addr = alloca i1
  %result = call i1 @llvm.target.has.feature(metadata !"avx2")
  store i1 %result, ptr %has_avx2.addr
  %loaded = load i1, ptr %has_avx2.addr
  br i1 %loaded, label %avx2.bb, label %fallback.bb

avx2.bb:
  call void @avx2_path()
  ret void

fallback.bb:
  call void @fallback_path()
  ret void
}

define void @test_o0_alloca_i8_bool() {
; HASWELL-LABEL: @test_o0_alloca_i8_bool(
; HASWELL-NOT:   @fallback_path
; HASWELL:       call void @avx2_path()
; HASWELL:       ret void
;
; GENERIC-LABEL: @test_o0_alloca_i8_bool(
; GENERIC-NOT:   @avx2_path
; GENERIC:       call void @fallback_path()
; GENERIC:       ret void
entry:
  %has_avx2.addr = alloca i8
  %result = call i1 @llvm.target.has.feature(metadata !"avx2")
  %ext = zext i1 %result to i8
  store i8 %ext, ptr %has_avx2.addr
  %loaded = load i8, ptr %has_avx2.addr
  %tobool = trunc i8 %loaded to i1
  br i1 %tobool, label %avx2.bb, label %fallback.bb

avx2.bb:
  call void @avx2_path()
  ret void

fallback.bb:
  call void @fallback_path()
  ret void
}

define void @test_o0_multi_load() {
; HASWELL-LABEL: @test_o0_multi_load(
; HASWELL-NOT:   @fallback_path
; HASWELL:       call void @avx2_path()
; HASWELL:       ret void
;
; GENERIC-LABEL: @test_o0_multi_load(
; GENERIC-NOT:   @avx2_path
; GENERIC:       call void @fallback_path()
; GENERIC:       ret void
entry:
  %has_avx2.addr = alloca i1
  %result = call i1 @llvm.target.has.feature(metadata !"avx2")
  store i1 %result, ptr %has_avx2.addr
  %loaded1 = load i1, ptr %has_avx2.addr
  br i1 %loaded1, label %check2, label %fallback.bb

check2:
  %loaded2 = load i1, ptr %has_avx2.addr
  br i1 %loaded2, label %avx2.bb, label %fallback.bb

avx2.bb:
  call void @avx2_path()
  ret void

fallback.bb:
  call void @fallback_path()
  ret void
}
