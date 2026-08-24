; s_cbranch_cdbgsys_or_user and s_trap are scalar and ignore EXEC, so divergent
; control flow around them needs an EXEC guard.
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 < %s | FileCheck %s --check-prefixes=GCN,DBGSTATUS
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1150 -O2 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefixes=GCN,DBGSTATUS
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O2 < %s | FileCheck %s --check-prefixes=GCN,DBGPRIV
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O2 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefixes=GCN,DBGPRIV
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 < %s | FileCheck %s --check-prefixes=GCN,DBGSTATUS
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O2 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefixes=GCN,DBGSTATUS
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 < %s | FileCheck %s --check-prefix=FUSED
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O0 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefix=FUSED
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O1 < %s | FileCheck %s --check-prefix=FUSED
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1310 -O1 -global-isel -global-isel-abort=1 < %s | FileCheck %s --check-prefix=FUSED

declare noundef i1 @llvm.is.debugging.enabled()
declare i1 @llvm.expect.i1(i1, i1 immarg)
declare void @llvm.debugtrap()
declare i32 @llvm.amdgcn.workitem.id.x()

; Nested divergent regions: each contributes a guard above the fused branch.
define amdgpu_kernel void @nested_divergent_debugtrap(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %outer = icmp ult i32 %tid, 8
  br i1 %outer, label %outer.region, label %exit

outer.region:
  store volatile i32 1, ptr addrspace(1) %out
  %inner = icmp eq i32 %tid, 3
  br i1 %inner, label %inner.region, label %outer.after

inner.region:
  %e = call i1 @llvm.is.debugging.enabled()
  br i1 %e, label %dbg, label %inner.after

dbg:
  call void @llvm.debugtrap()
  br label %inner.after

inner.after:
  store volatile i32 2, ptr addrspace(1) %out
  br label %outer.after

outer.after:
  store volatile i32 3, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}

; GCN-LABEL: nested_divergent_debugtrap:
; GCN: s_cbranch_execz [[OUTER_SKIP:.LBB[0-9_]+]]
; GCN: s_cbranch_execz [[INNER_SKIP:.LBB[0-9_]+]]
; GCN: s_cbranch_cdbgsys_or_user [[NESTED_DEBUG:.LBB[0-9_]+]]
; GCN: [[NESTED_DEBUG]]:
; GCN-NEXT: s_trap 3
; GCN: [[INNER_SKIP]]:
; GCN: [[OUTER_SKIP]]:

; O0/O1 fuse the single-use loop query; O2 materializes it after CFG restructuring.
define amdgpu_kernel void @loop_debugtrap_exec_guard(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %d = icmp ult i32 %tid, 8
  br i1 %d, label %L, label %E

L:
  %e = call i1 @llvm.is.debugging.enabled()
  br i1 %e, label %dbg, label %M

dbg:
  call void @llvm.debugtrap()
  br label %M

M:
  store i32 2, ptr addrspace(1) %out
  br i1 %d, label %L, label %E

E:
  ret void
}

; GCN-LABEL: loop_debugtrap_exec_guard:
; GCN: s_cbranch_execz [[LOOP_SKIP:.LBB[0-9_]+]]
; GCN: [[LOOP_HDR:.LBB[0-9_]+]]:
; DBGSTATUS: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_{{(WAVE_)?}}STATUS, 20, 2)
; DBGPRIV: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_WAVE_STATE_PRIV, 16, 2)
; GCN-NOT: s_cbranch_cdbgsys_or_user
; GCN: s_trap 3
; GCN: [[LOOP_SKIP]]:

; FUSED-LABEL: loop_debugtrap_exec_guard:
; FUSED: s_cbranch_execz
; FUSED-NOT: s_getreg_b32
; FUSED: s_cbranch_cdbgsys_or_user
; FUSED-NOT: s_getreg_b32

define amdgpu_kernel void @divergent_debug_break() {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %divergent = icmp eq i32 %id, 0
  br i1 %divergent, label %query, label %normal

query:
  %raw = call i1 @llvm.is.debugging.enabled()
  %enabled = call i1 @llvm.expect.i1(i1 %raw, i1 false)
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.debugtrap()
  br label %normal

normal:
  ret void
}

; GCN-LABEL: divergent_debug_break:
; GCN: s_cbranch_execz [[DIVERGENT_NORMAL:.LBB[0-9_]+]]
; GCN-COUNT-1: s_cbranch_cdbgsys_or_user [[DIVERGENT_DEBUG:.LBB[0-9_]+]]
; GCN-NOT: s_cbranch_cdbgsys_or_user
; GCN: [[DIVERGENT_DEBUG]]:
; GCN-NEXT: s_trap 3
; GCN: [[DIVERGENT_NORMAL]]:{{.*}}%normal

define amdgpu_kernel void @two_query_sites(ptr addrspace(1) %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %choose.left = icmp eq i32 %id, 0
  br i1 %choose.left, label %left, label %right

left:
  %left.enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %left.enabled, label %left.debug, label %join

left.debug:
  store volatile i32 1, ptr addrspace(1) %out
  br label %join

right:
  %right.enabled = call i1 @llvm.is.debugging.enabled()
  br i1 %right.enabled, label %right.debug, label %join

right.debug:
  store volatile i32 2, ptr addrspace(1) %out
  br label %join

join:
  ret void
}

; GCN-LABEL: two_query_sites:
; GCN-COUNT-2: s_cbranch_cdbgsys_or_user

; Uniform query outside a divergent trap: only the inner branch needs a guard.
define amdgpu_kernel void @debug_enabled_then_divergent_trap(ptr addrspace(1) %out) {
entry:
  %e = call i1 @llvm.is.debugging.enabled()
  br i1 %e, label %dbg.region, label %exit

dbg.region:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gt8 = icmp ugt i32 %tid, 8
  br i1 %gt8, label %trap, label %after

trap:
  call void @llvm.debugtrap()
  br label %after

after:
  store volatile i32 1, ptr addrspace(1) %out
  br label %exit

exit:
  ret void
}

; GCN-LABEL: debug_enabled_then_divergent_trap:
; GCN: s_cbranch_cdbgsys_or_user [[DBG_REGION:.LBB[0-9_]+]]
; GCN: [[DBG_REGION]]:
; GCN: s_cbranch_execz [[TRAP_SKIP:.LBB[0-9_]+]]
; GCN: s_trap 3
; GCN: [[TRAP_SKIP]]:

; Test an out-of-line DebugBreak helper from uniform and divergent call sites.
define void @debugbreak_helper() noinline {
entry:
  %raw = call i1 @llvm.is.debugging.enabled()
  %enabled = call i1 @llvm.expect.i1(i1 %raw, i1 false)
  br i1 %enabled, label %debug, label %normal

debug:
  call void @llvm.debugtrap()
  br label %normal

normal:
  ret void
}

; GCN-LABEL: debugbreak_helper:
; GCN-NOT: s_cbranch_execz
; GCN: s_cbranch_cdbgsys_or_user [[HELPER_DEBUG:.LBB[0-9_]+]]
; GCN-NOT: s_cbranch_cdbgsys_or_user
; GCN: [[HELPER_DEBUG]]:
; GCN-NEXT: s_trap 3

define amdgpu_kernel void @uniform_helper_call() {
entry:
  call void @debugbreak_helper()
  ret void
}

; GCN-LABEL: uniform_helper_call:
; GCN-NOT: s_cbranch_execz
; GCN: debugbreak_helper
; GCN: s_{{swappc_b64|swap_pc_i64}}

define amdgpu_kernel void @divergent_helper_call() {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %divergent = icmp eq i32 %id, 0
  br i1 %divergent, label %call, label %normal

call:
  call void @debugbreak_helper()
  br label %normal

normal:
  ret void
}

; GCN-LABEL: divergent_helper_call:
; GCN: s_cbranch_execz [[CALL_SKIP:.LBB[0-9_]+]]
; GCN: debugbreak_helper
; GCN: s_{{swappc_b64|swap_pc_i64}}
; GCN: [[CALL_SKIP]]:{{.*}}%normal
