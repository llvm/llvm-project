; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target datalayout = "A5"

declare void @llvm.amdgcn.kill(i1)
declare void @llvm.amdgcn.wqm.demote(i1)

; CHECK: Intrinsic can only be used from functions with the amdgpu_ps calling convention
; CHECK-NEXT: call void @llvm.amdgcn.kill(i1 true)
define amdgpu_cs void @cs_kill() {
  call void @llvm.amdgcn.kill(i1 true)
  ret void
}

; CHECK: Intrinsic can only be used from functions with the amdgpu_ps calling convention
; CHECK-NEXT: call void @llvm.amdgcn.kill(i1 true)
define amdgpu_gs void @gs_kill() {
  call void @llvm.amdgcn.kill(i1 true)
  ret void
}

; CHECK: Intrinsic can only be used from functions with the amdgpu_ps calling convention
; CHECK-NEXT: call void @llvm.amdgcn.wqm.demote(i1 true)
define amdgpu_cs void @cs_wqm_demote() {
  call void @llvm.amdgcn.wqm.demote(i1 true)
  ret void
}

; CHECK: Intrinsic can only be used from functions with the amdgpu_ps calling convention
; CHECK-NEXT: call void @llvm.amdgcn.wqm.demote(i1 true)
define amdgpu_gs void @gs_wqm_demote() {
  call void @llvm.amdgcn.wqm.demote(i1 true)
  ret void
}
