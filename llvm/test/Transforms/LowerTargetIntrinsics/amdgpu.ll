; RUN: opt -S -passes=lower-target-intrinsics -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 < %s | FileCheck %s --check-prefix=GFX1030
; RUN: opt -S -passes=lower-target-intrinsics -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck %s --check-prefix=GFX90A

declare void @rdna_path()
declare void @cdna_path()
declare void @generic_path()

define void @test_amdgpu_cpu_dispatch() {
; GFX1030-LABEL: @test_amdgpu_cpu_dispatch(
; GFX1030-NOT:   @cdna_path
; GFX1030-NOT:   @generic_path
; GFX1030:       call void @rdna_path()
; GFX1030:       ret void
;
; GFX90A-LABEL: @test_amdgpu_cpu_dispatch(
; GFX90A-NOT:   @rdna_path
; GFX90A-NOT:   @generic_path
; GFX90A:       call void @cdna_path()
; GFX90A:       ret void
entry:
  %is_gfx1030 = call i1 @llvm.target.is.cpu(metadata !"gfx1030")
  br i1 %is_gfx1030, label %rdna, label %check_cdna

rdna:
  call void @rdna_path()
  ret void

check_cdna:
  %is_gfx90a = call i1 @llvm.target.is.cpu(metadata !"gfx90a")
  br i1 %is_gfx90a, label %cdna, label %generic

cdna:
  call void @cdna_path()
  ret void

generic:
  call void @generic_path()
  ret void
}
