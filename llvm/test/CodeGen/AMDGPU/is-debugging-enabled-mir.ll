; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=irtranslator %s -o - | FileCheck %s --check-prefix=TRANSLATE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=legalizer %s -o - | FileCheck %s --check-prefix=FALSE --implicit-check-not=G_IS_DEBUGGING_ENABLED --implicit-check-not=amdgcn.s.getreg --implicit-check-not=S_CBRANCH_CDBG
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=legalizer %s -o - | FileCheck %s --check-prefix=CUSTOM --implicit-check-not=G_IS_DEBUGGING_ENABLED --implicit-check-not=S_CBRANCH_CDBG
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=SELECTED --implicit-check-not=S_GETREG_B32
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -global-isel=1 -global-isel-abort=1 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=SELECTED --implicit-check-not=S_GETREG_B32

declare i1 @llvm.is.debugging.enabled()
declare void @llvm.debugtrap()

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

define amdgpu_kernel void @branch_query() {
; TRANSLATE-LABEL: name: branch_query
; TRANSLATE: [[COND:%[0-9]+]]:_(i1) = nomerge G_IS_DEBUGGING_ENABLED
; TRANSLATE: G_BRCOND [[COND]](i1), %bb.
; FALSE-LABEL: name: branch_query
; FALSE: G_BR %bb.
; CUSTOM-LABEL: name: branch_query
; CUSTOM: [[BITS:%[0-9]+]]:_(s32) = nomerge G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.s.getreg)
; CUSTOM: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CUSTOM: [[COND:%[0-9]+]]:_(i1) = G_ICMP intpred(ne), [[BITS]](s32), [[ZERO]]
; CUSTOM: G_BRCOND {{%[0-9]+}}(i1), %bb.
; SELECTED-LABEL: name: branch_query
; SELECTED: nomerge S_CBRANCH_CDBGSYS_OR_USER %bb.
entry:
  %enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %enabled, label %debug, label %exit

debug:
  call void @llvm.debugtrap()
  br label %exit

exit:
  ret void
}
