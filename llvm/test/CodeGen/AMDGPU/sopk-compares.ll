; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; Since this intrinsic is exposed as a constant after isel, use it to
; defeat the DAG's compare with constant canonicalizations.
declare i32 @llvm.amdgcn.groupstaticsize() #1

@lds = addrspace(3) global [512 x i32] undef, align 4

; GCN-LABEL: {{^}}br_scc_eq_i32_inline_imm:
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 4{{$}}
define amdgpu_kernel void @br_scc_eq_i32_inline_imm(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, 4
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i32_simm16_max:
; GCN: s_cmpk_eq_i32 s{{[0-9]+}}, 0x7fff{{$}}
define amdgpu_kernel void @br_scc_eq_i32_simm16_max(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, 32767
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i32_simm16_max_p1:
; GCN: s_cmpk_eq_u32 s{{[0-9]+}}, 0x8000{{$}}
define amdgpu_kernel void @br_scc_eq_i32_simm16_max_p1(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, 32768
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ne_i32_simm16_max_p1:
; GCN: s_cmpk_lg_u32 s{{[0-9]+}}, 0x8000{{$}}
define amdgpu_kernel void @br_scc_ne_i32_simm16_max_p1(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp ne i32 %cond, 32768
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i32_simm16_min:
; GCN: s_cmpk_eq_i32 s{{[0-9]+}}, 0x8000{{$}}
define amdgpu_kernel void @br_scc_eq_i32_simm16_min(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, -32768
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i32_simm16_min_m1:
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 0xffff7fff{{$}}
define amdgpu_kernel void @br_scc_eq_i32_simm16_min_m1(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, -32769
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i32_uimm15_max:
; GCN: s_cmpk_eq_u32 s{{[0-9]+}}, 0xffff{{$}}
define amdgpu_kernel void @br_scc_eq_i32_uimm15_max(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, 65535
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i32_uimm16_max:
; GCN: s_cmpk_eq_u32 s{{[0-9]+}}, 0xffff{{$}}
define amdgpu_kernel void @br_scc_eq_i32_uimm16_max(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, 65535
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i32_uimm16_max_p1:
; GCN: s_cmp_eq_u32 s{{[0-9]+}}, 0x10000{{$}}
define amdgpu_kernel void @br_scc_eq_i32_uimm16_max_p1(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, 65536
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}br_scc_eq_i32:
; GCN: s_cmpk_eq_i32 s{{[0-9]+}}, 0x41{{$}}
define amdgpu_kernel void @br_scc_eq_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i32 %cond, 65
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ne_i32:
; GCN: s_cmpk_lg_i32 s{{[0-9]+}}, 0x41{{$}}
define amdgpu_kernel void @br_scc_ne_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp ne i32 %cond, 65
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_sgt_i32:
; GCN: s_cmpk_gt_i32 s{{[0-9]+}}, 0x41{{$}}
define amdgpu_kernel void @br_scc_sgt_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp sgt i32 %cond, 65
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_sgt_i32_simm16_max:
; GCN: s_cmpk_gt_i32 s{{[0-9]+}}, 0x7fff{{$}}
define amdgpu_kernel void @br_scc_sgt_i32_simm16_max(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp sgt i32 %cond, 32767
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_sgt_i32_simm16_max_p1:
; GCN: s_cmp_gt_i32 s{{[0-9]+}}, 0x8000{{$}}
define amdgpu_kernel void @br_scc_sgt_i32_simm16_max_p1(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp sgt i32 %cond, 32768
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_sge_i32:
; GCN: s_cmpk_ge_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @br_scc_sge_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp sge i32 %cond, %size
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_slt_i32:
; GCN: s_cmpk_lt_i32 s{{[0-9]+}}, 0x41{{$}}
define amdgpu_kernel void @br_scc_slt_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp slt i32 %cond, 65
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_sle_i32:
; GCN: s_cmpk_le_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @br_scc_sle_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp sle i32 %cond, %size
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ugt_i32:
; GCN: s_cmpk_gt_u32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @br_scc_ugt_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp ugt i32 %cond, %size
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_uge_i32:
; GCN: s_cmpk_ge_u32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @br_scc_uge_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp uge i32 %cond, %size
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ult_i32:
; GCN: s_cmpk_lt_u32 s{{[0-9]+}}, 0x41{{$}}
define amdgpu_kernel void @br_scc_ult_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp ult i32 %cond, 65
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ult_i32_min_simm16:
; GCN: s_cmp_lt_u32 s2, 0xffff8000
define amdgpu_kernel void @br_scc_ult_i32_min_simm16(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp ult i32 %cond, -32768
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ult_i32_min_simm16_m1:
; GCN: s_cmp_lt_u32 s{{[0-9]+}}, 0xffff7fff{{$}}
define amdgpu_kernel void @br_scc_ult_i32_min_simm16_m1(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp ult i32 %cond, -32769
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ule_i32:
; GCN: s_cmpk_le_u32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @br_scc_ule_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp ule i32 %cond, %size
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_eq_i32:
; GCN: s_cmpk_eq_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_eq_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp eq i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_ne_i32:
; GCN: s_cmpk_lg_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_ne_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp ne i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_sgt_i32:
; GCN: s_cmpk_lt_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_sgt_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp sgt i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_sge_i32:
; GCN: s_cmpk_le_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_sge_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp sge i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_slt_i32:
; GCN: s_cmpk_gt_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_slt_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp slt i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_sle_i32:
; GCN: s_cmpk_ge_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_sle_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp sle i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_ugt_i32:
; GCN: s_cmpk_lt_u32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_ugt_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp ugt i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_uge_i32:
; GCN: s_cmpk_le_u32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_uge_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp uge i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_ult_i32:
; GCN: s_cmpk_gt_u32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_ult_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp ult i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_br_scc_ule_i32:
; GCN: s_cmpk_ge_u32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_br_scc_ule_i32(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %cmp0 = icmp ule i32 %size, %cond
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ult_i32_non_u16:
; GCN: s_cmp_lt_u32 s2, 0xfffff7ff
define amdgpu_kernel void @br_scc_ult_i32_non_u16(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %not.size = xor i32 %size, -1
  %cmp0 = icmp ult i32 %cond, %not.size
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "; $0", "v"([512 x i32] addrspace(3)* @lds)
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i64_inline_imm:
; VI: s_cmp_eq_u64 s{{\[[0-9]+:[0-9]+\]}}, 4

; SI: v_cmp_eq_u64_e64
define amdgpu_kernel void @br_scc_eq_i64_inline_imm(i64 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i64 %cond, 4
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_eq_i64_simm16:
; VI-DAG: s_movk_i32 s[[K_LO:[0-9]+]], 0x4d2
; VI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 1
; VI: s_cmp_eq_u64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}

; SI: v_cmp_eq_u64_e32
define amdgpu_kernel void @br_scc_eq_i64_simm16(i64 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp eq i64 %cond, 4294968530
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ne_i64_inline_imm:
; VI: s_cmp_lg_u64 s{{\[[0-9]+:[0-9]+\]}}, 4

; SI: v_cmp_ne_u64_e64
define amdgpu_kernel void @br_scc_ne_i64_inline_imm(i64 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp ne i64 %cond, 4
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}br_scc_ne_i64_simm16:
; VI-DAG: s_movk_i32 s[[K_LO:[0-9]+]], 0x4d2
; VI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 1
; VI: s_cmp_lg_u64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}

; SI: v_cmp_ne_u64_e32
define amdgpu_kernel void @br_scc_ne_i64_simm16(i64 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %cmp0 = icmp ne i64 %cond, 4294968530
  br i1 %cmp0, label %endif, label %if

if:
  call void asm sideeffect "", ""()
  br label %endif

endif:
  store volatile i32 1, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
