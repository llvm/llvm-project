; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1150 -mattr=+trap-handler -global-isel=0 < %s | FileCheck %s --check-prefix=PAL
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1150 -mattr=+trap-handler -global-isel=1 -global-isel-abort=1 < %s | FileCheck %s --check-prefix=PAL
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1310 -mattr=+trap-handler -global-isel=0 < %s | FileCheck %s --check-prefix=PAL
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1310 -mattr=+trap-handler -global-isel=1 -global-isel-abort=1 < %s | FileCheck %s --check-prefix=PAL
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1150 -global-isel=0 < %s 2>&1 | FileCheck %s --check-prefix=NO-TRAP
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1150 -mattr=-trap-handler -global-isel=1 -global-isel-abort=1 < %s 2>&1 | FileCheck %s --check-prefix=NO-TRAP

declare void @llvm.debugtrap() #0
declare void @llvm.trap() #1
declare i32 @llvm.amdgcn.workitem.id.x() #2

; PAL-LABEL: direct_debugtrap:
; PAL: s_trap 3
; PAL: global_store_b32
define amdgpu_kernel void @direct_debugtrap(ptr addrspace(1) %out) {
entry:
  call void @llvm.debugtrap()
  store volatile i32 1, ptr addrspace(1) %out
  ret void
}

; PAL-LABEL: divergent_debugtrap:
; PAL: s_cbranch_execz [[SKIP:.LBB[0-9_]+]]
; PAL: s_trap 3
; PAL: [[SKIP]]:
define amdgpu_kernel void @divergent_debugtrap() {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %divergent = icmp eq i32 %id, 0
  br i1 %divergent, label %trap, label %exit

trap:
  call void @llvm.debugtrap()
  br label %exit

exit:
  ret void
}

; PAL-LABEL: ordinary_trap:
; PAL-NOT: s_trap 2
; PAL: s_endpgm
define amdgpu_kernel void @ordinary_trap() {
entry:
  call void @llvm.trap()
  unreachable
}

; NO-TRAP-COUNT-2: warning: {{.*}}debugtrap handler not supported
; NO-TRAP-NOT: s_trap

attributes #0 = { nounwind }
attributes #1 = { nounwind noreturn }
attributes #2 = { nounwind readnone speculatable }
