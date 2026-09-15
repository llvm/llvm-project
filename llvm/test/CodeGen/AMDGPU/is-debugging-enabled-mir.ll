; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=irtranslator %s -o - | FileCheck %s --check-prefix=TRANSLATE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=legalizer %s -o - | FileCheck %s --check-prefix=FALSE --implicit-check-not=G_IS_DEBUGGING_ENABLED --implicit-check-not=amdgcn.s.getreg --implicit-check-not=S_CBRANCH_CDBG
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=legalizer %s -o - | FileCheck %s --check-prefix=CUSTOM --implicit-check-not=G_IS_DEBUGGING_ENABLED
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=SELECTED
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=SELECTED

declare i1 @llvm.is.debugging.enabled()

define amdgpu_kernel void @two_queries(ptr addrspace(1) %out) {
; TRANSLATE-LABEL: name: two_queries
; TRANSLATE-COUNT-2: {{%[0-9]+}}:_(i1) = nomerge G_IS_DEBUGGING_ENABLED
; FALSE-LABEL: name: two_queries
; FALSE: {{%[0-9]+}}:_(i32) = G_CONSTANT i32 0
; CUSTOM-LABEL: name: two_queries
; CUSTOM-COUNT-2: nomerge G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.s.getreg)
; SELECTED-LABEL: name: two_queries
; SELECTED-COUNT-2: nomerge S_GETREG_B32
  %first = call i1 @llvm.is.debugging.enabled()
  %a = zext i1 %first to i32
  store volatile i32 %a, ptr addrspace(1) %out
  %second = call i1 @llvm.is.debugging.enabled()
  %b = zext i1 %second to i32
  store volatile i32 %b, ptr addrspace(1) %out
  ret void
}
