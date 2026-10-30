; RUN: llc -mtriple=amdgpu10.30-amd-amdhsa -mcpu=gfx1030 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mcpu=gfx1100 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=amdgpu11-amd-amdhsa -mcpu=gfx11-generic -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=amdgpu11.70-amd-amdhsa -mcpu=gfx1170 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=amdgpu11.50-amd-amdhsa -mcpu=gfx1150 -mattr=-debugging-enabled-query -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mcpu=gfx1100 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs %s -o - | FileCheck %s

declare noundef i1 @llvm.is.debugging.enabled()

define i1 @unsupported_subtarget() {
; CHECK-LABEL: unsupported_subtarget:
; CHECK-NOT: s_getreg
; CHECK-NOT: s_cbranch_cdbg
; CHECK: v_mov_b32_e32 v0, 0
  %enabled = call i1 @llvm.is.debugging.enabled()
  ret i1 %enabled
}
