; RUN: llc -global-isel=0 -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=HSA-TRAP %s
; RUN: llc -global-isel=1 -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=HSA-TRAP %s

; RUN: llc -global-isel=0 -mtriple=amdgcn--amdhsa -mattr=+trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=HSA-TRAP %s
; RUN: llc -global-isel=1 -mtriple=amdgcn--amdhsa -mattr=+trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=HSA-TRAP %s
; RUN: llc -global-isel=0 -mtriple=amdgcn--amdhsa -mattr=-trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NO-HSA-TRAP %s
; RUN: llc -global-isel=1 -mtriple=amdgcn--amdhsa -mattr=-trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NO-HSA-TRAP %s
; RUN: llc -global-isel=0 -mtriple=amdgcn--amdhsa -mattr=-trap-handler -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING %s
; RUN: llc -global-isel=1 -mtriple=amdgcn--amdhsa -mattr=-trap-handler -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING %s

; enable trap handler feature
; RUN: llc -global-isel=0 -mtriple=amdgcn-unknown-mesa3d -mattr=+trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NO-MESA-TRAP -check-prefix=TRAP-BIT -check-prefix=MESA-TRAP %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-unknown-mesa3d -mattr=+trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NO-MESA-TRAP -check-prefix=TRAP-BIT -check-prefix=MESA-TRAP %s
; RUN: llc -global-isel=0 -mtriple=amdgcn-unknown-mesa3d -mattr=+trap-handler -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -check-prefix=TRAP-BIT %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-unknown-mesa3d -mattr=+trap-handler -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -check-prefix=TRAP-BIT %s

; disable trap handler feature
; RUN: llc -global-isel=0 -mtriple=amdgcn-unknown-mesa3d -mattr=-trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NO-MESA-TRAP -check-prefix=NO-TRAP-BIT -check-prefix=NOMESA-TRAP %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-unknown-mesa3d -mattr=-trap-handler -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=NO-MESA-TRAP -check-prefix=NO-TRAP-BIT -check-prefix=NOMESA-TRAP %s
; RUN: llc -global-isel=0 -mtriple=amdgcn-unknown-mesa3d -mattr=-trap-handler -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -check-prefix=NO-TRAP-BIT %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-unknown-mesa3d -mattr=-trap-handler -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING -check-prefix=NO-TRAP-BIT %s

; RUN: llc -global-isel=0 -mtriple=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=GCN-WARNING %s

; GCN-WARNING: warning: <unknown>:0:0: in function hsa_debugtrap void (ptr addrspace(1)): debugtrap handler not supported


declare void @llvm.trap() #0
declare void @llvm.debugtrap() #1

; MESA-TRAP: .section .AMDGPU.config
; MESA-TRAP:  .long   47180
; MESA-TRAP-NEXT: .long   208

; NOMESA-TRAP: .section .AMDGPU.config
; NOMESA-TRAP:  .long   47180
; NOMESA-TRAP-NEXT: .long   144

; GCN-LABEL: {{^}}hsa_trap:
; HSA-TRAP: s_mov_b64 s[0:1], s[4:5]
; HSA-TRAP: s_trap 2
; HSA-TRAP: COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0

; for llvm.trap in hsa path without ABI, direct generate s_endpgm instruction without any warning information
; NO-HSA-TRAP: s_endpgm
; NO-HSA-TRAP: COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0

; TRAP-BIT: enable_trap_handler = 1
; NO-TRAP-BIT: enable_trap_handler = 0
; NO-MESA-TRAP: s_endpgm
define amdgpu_kernel void @hsa_trap(ptr addrspace(1) nocapture readonly %arg0) {
  store volatile i32 1, ptr addrspace(1) %arg0
  call void @llvm.trap()
  unreachable
  store volatile i32 2, ptr addrspace(1) %arg0
  ret void
}

; MESA-TRAP: .section .AMDGPU.config
; MESA-TRAP:  .long   47180
; MESA-TRAP-NEXT: .long   204

; NOMESA-TRAP: .section .AMDGPU.config
; NOMESA-TRAP:  .long   47180
; NOMESA-TRAP-NEXT: .long   140

; GCN-LABEL: {{^}}hsa_debugtrap:
; HSA-TRAP: s_trap 3
; HSA-TRAP: flat_store_dword v[0:1], v3
; HSA-TRAP: COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0

; for llvm.debugtrap in non-hsa path without ABI, generate a warning and a s_endpgm instruction
; NO-HSA-TRAP: s_endpgm

; TRAP-BIT: enable_trap_handler = 1
; NO-TRAP-BIT: enable_trap_handler = 0
; NO-MESA-TRAP: s_endpgm
define amdgpu_kernel void @hsa_debugtrap(ptr addrspace(1) nocapture readonly %arg0) {
  store volatile i32 1, ptr addrspace(1) %arg0
  call void @llvm.debugtrap()
  store volatile i32 2, ptr addrspace(1) %arg0
  ret void
}

; For non-HSA path
; GCN-LABEL: {{^}}trap:
; TRAP-BIT: enable_trap_handler = 1
; NO-TRAP-BIT: enable_trap_handler = 0
; NO-HSA-TRAP: s_endpgm
; NO-MESA-TRAP: s_endpgm
define amdgpu_kernel void @trap(ptr addrspace(1) nocapture readonly %arg0) {
  store volatile i32 1, ptr addrspace(1) %arg0
  call void @llvm.trap()
  unreachable
  store volatile i32 2, ptr addrspace(1) %arg0
  ret void
}

; GCN-LABEL: {{^}}non_entry_trap:
; TRAP-BIT: enable_trap_handler = 1
; NO-TRAP-BIT: enable_trap_handler = 0

; HSA-TRAP: BB{{[0-9]_[0-9]+}}: ; %trap
; HSA-TRAP: s_mov_b64 s[0:1], s[4:5]
; HSA-TRAP-NEXT: s_trap 2
define amdgpu_kernel void @non_entry_trap(ptr addrspace(1) nocapture readonly %arg0) local_unnamed_addr {
entry:
  %tmp29 = load volatile i32, ptr addrspace(1) %arg0
  %cmp = icmp eq i32 %tmp29, -1
  br i1 %cmp, label %ret, label %trap

trap:
  call void @llvm.trap()
  unreachable

ret:
  store volatile i32 3, ptr addrspace(1) %arg0
  ret void
}

; GCN-LABEL: {{^}}non_entry_trap_no_unreachable:
; TRAP-BIT: enable_trap_handler = 1
; NO-TRAP-BIT: enable_trap_handler = 0

; HSA-TRAP: BB{{[0-9]_[0-9]+}}: ; %trap
; HSA-TRAP: s_mov_b64 s[0:1], s[4:5]
; HSA-TRAP-NEXT: s_trap 2
define amdgpu_kernel void @non_entry_trap_no_unreachable(ptr addrspace(1) nocapture readonly %arg0) local_unnamed_addr {
entry:
  %tmp29 = load volatile i32, ptr addrspace(1) %arg0
  %cmp = icmp eq i32 %tmp29, -1
  br i1 %cmp, label %ret, label %trap

trap:
  call void @llvm.trap()
  store volatile i32 1234, ptr addrspace(3) null
  br label %ret

ret:
  store volatile i32 3, ptr addrspace(1) %arg0
  ret void
}

attributes #0 = { nounwind noreturn }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
