; REQUIRES: x86-registered-target
; RUN: llc -mtriple=x86_64-unknown-linux -mcpu=haswell -O0 -o - %s | FileCheck %s --check-prefix=HASWELL
; RUN: llc -mtriple=x86_64-unknown-linux -mcpu=haswell -O2 -o - %s | FileCheck %s --check-prefix=HASWELL
; RUN: llc -mtriple=x86_64-unknown-linux -mcpu=x86-64 -O0 -o - %s | FileCheck %s --check-prefix=GENERIC
; RUN: llc -mtriple=x86_64-unknown-linux -mcpu=x86-64 -O2 -o - %s | FileCheck %s --check-prefix=GENERIC

; Verify the intrinsics are resolved in the backend pipeline.

declare void @avx2_path()
declare void @fallback_path()

define void @test_pipeline_o0() {
; HASWELL-LABEL: test_pipeline_o0:
; HASWELL:       callq avx2_path
; HASWELL-NOT:   callq fallback_path
;
; GENERIC-LABEL: test_pipeline_o0:
; GENERIC:       callq fallback_path
; GENERIC-NOT:   callq avx2_path
entry:
  %has_avx2 = call i1 @llvm.target.has.feature(metadata !"avx2")
  br i1 %has_avx2, label %avx2.bb, label %fallback.bb

avx2.bb:
  call void @avx2_path()
  ret void

fallback.bb:
  call void @fallback_path()
  ret void
}

; O0 with alloca pattern
define void @test_pipeline_o0_alloca() {
; HASWELL-LABEL: test_pipeline_o0_alloca:
; HASWELL:       callq avx2_path
; HASWELL-NOT:   callq fallback_path
;
; GENERIC-LABEL: test_pipeline_o0_alloca:
; GENERIC:       callq fallback_path
; GENERIC-NOT:   callq avx2_path
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

declare i1 @llvm.target.has.feature(metadata)
