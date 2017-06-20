; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=1 -march=amdgcn -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck -check-prefix=TOVGPR -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=1 -amdgpu-spill-sgpr-to-smem=0 -march=amdgcn -mcpu=tonga -mattr=+vgpr-spilling  -verify-machineinstrs < %s | FileCheck -check-prefix=TOVGPR -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=0 -march=amdgcn -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck -check-prefix=TOVMEM -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=0 -amdgpu-spill-sgpr-to-smem=0 -march=amdgcn -mcpu=tonga -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck -check-prefix=TOVMEM -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=0 -amdgpu-spill-sgpr-to-smem=1 -march=amdgcn -mcpu=tonga -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck -check-prefix=TOSMEM -check-prefix=GCN %s

; XXX - Why does it like to use vcc?

; GCN-LABEL: {{^}}spill_m0:
; TOSMEM: s_mov_b32 s[[LO:[0-9]+]], SCRATCH_RSRC_DWORD0
; TOSMEM: s_mov_b32 s[[HI:[0-9]+]], 0xe80000

; GCN-DAG: s_cmp_lg_u32

; TOVGPR-DAG: s_mov_b32 [[M0_COPY:s[0-9]+]], m0
; TOVGPR: v_writelane_b32 [[SPILL_VREG:v[0-9]+]], [[M0_COPY]], 0

; TOVMEM-DAG: s_mov_b32 [[M0_COPY:s[0-9]+]], m0
; TOVMEM-DAG: v_mov_b32_e32 [[SPILL_VREG:v[0-9]+]], [[M0_COPY]]
; TOVMEM: buffer_store_dword [[SPILL_VREG]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:4 ; 4-byte Folded Spill

; TOSMEM-DAG: s_mov_b32 [[M0_COPY:s[0-9]+]], m0
; TOSMEM: s_add_u32 m0, s3, 0x100{{$}}
; TOSMEM-NOT: [[M0_COPY]]
; TOSMEM: s_buffer_store_dword [[M0_COPY]], s{{\[}}[[LO]]:[[HI]]], m0 ; 4-byte Folded Spill

; GCN: s_cbranch_scc1 [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: [[ENDIF]]:
; TOVGPR: v_readlane_b32 [[M0_RESTORE:s[0-9]+]], [[SPILL_VREG]], 0
; TOVGPR: s_mov_b32 m0, [[M0_RESTORE]]

; TOVMEM: buffer_load_dword [[RELOAD_VREG:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:4 ; 4-byte Folded Reload
; TOVMEM: s_waitcnt vmcnt(0)
; TOVMEM: v_readfirstlane_b32 [[M0_RESTORE:s[0-9]+]], [[RELOAD_VREG]]
; TOVMEM: s_mov_b32 m0, [[M0_RESTORE]]

; TOSMEM: s_add_u32 m0, s3, 0x100{{$}}
; TOSMEM: s_buffer_load_dword [[M0_RESTORE:s[0-9]+]], s{{\[}}[[LO]]:[[HI]]], m0 ; 4-byte Folded Reload
; TOSMEM-NOT: [[M0_RESTORE]]
; TOSMEM: s_mov_b32 m0, [[M0_RESTORE]]

; GCN: s_add_i32 s{{[0-9]+}}, m0, 1
define amdgpu_kernel void @spill_m0(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %m0 = call i32 asm sideeffect "s_mov_b32 m0, 0", "={M0}"() #0
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %endif

if:
  call void asm sideeffect "v_nop", ""() #0
  br label %endif

endif:
  %foo = call i32 asm sideeffect "s_add_i32 $0, $1, 1", "=s,{M0}"(i32 %m0) #0
  store i32 %foo, i32 addrspace(1)* %out
  ret void
}

@lds = internal addrspace(3) global [64 x float] undef

; m0 is killed, so it isn't necessary during the entry block spill to preserve it
; GCN-LABEL: {{^}}spill_kill_m0_lds:
; GCN: s_mov_b32 m0, s6
; GCN: v_interp_mov_f32

; TOSMEM-NOT: s_m0
; TOSMEM: s_add_u32 m0, s7, 0x100
; TOSMEM-NEXT: s_buffer_store_dword s{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, m0 ; 4-byte Folded Spill
; FIXME: RegScavenger::isRegUsed() always returns true if m0 is reserved, so we have to save and restore it
; FIXME-TOSMEM-NOT: m0

; FIXME-TOSMEM-NOT: m0
; TOSMEM: s_add_u32 m0, s7, 0x200
; TOSMEM: s_buffer_store_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, m0 ; 8-byte Folded Spill
; FIXME-TOSMEM-NOT: m0

; TOSMEM: s_mov_b64 exec,
; TOSMEM: s_cbranch_execz
; TOSMEM: s_branch

; TOSMEM: BB{{[0-9]+_[0-9]+}}:
; TOSMEM: s_add_u32 m0, s7, 0x200
; TOSMEM-NEXT: s_buffer_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, m0 ; 8-byte Folded Reload


; GCN-NOT: v_readlane_b32 m0
; GCN-NOT: s_buffer_store_dword m0
; GCN-NOT: s_buffer_load_dword m0
define amdgpu_ps void @spill_kill_m0_lds(<16 x i8> addrspace(2)* inreg %arg, <16 x i8> addrspace(2)* inreg %arg1, <32 x i8> addrspace(2)* inreg %arg2, i32 inreg %m0) #0 {
main_body:
  %tmp = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %m0)
  %cmp = fcmp ueq float 0.000000e+00, %tmp
  br i1 %cmp, label %if, label %else

if:                                               ; preds = %main_body
  %lds_ptr = getelementptr [64 x float], [64 x float] addrspace(3)* @lds, i32 0, i32 0
  %lds_data = load float, float addrspace(3)* %lds_ptr
  br label %endif

else:                                             ; preds = %main_body
  %interp = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %m0)
  br label %endif

endif:                                            ; preds = %else, %if
  %export = phi float [ %lds_data, %if ], [ %interp, %else ]
  %tmp4 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %export, float %export)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %tmp4, <2 x half> %tmp4, i1 true, i1 true) #0
  ret void
}

; Force save and restore of m0 during SMEM spill
; GCN-LABEL: {{^}}m0_unavailable_spill:

; GCN: ; def m0, 1

; GCN: s_mov_b32 m0, s2
; GCN: v_interp_mov_f32

; GCN: ; clobber m0

; TOSMEM: s_mov_b32 s2, m0
; TOSMEM: s_add_u32 m0, s3, 0x100
; TOSMEM-NEXT: s_buffer_store_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, m0 ; 8-byte Folded Spill
; TOSMEM: s_mov_b32 m0, s2

; TOSMEM: s_mov_b64 exec,
; TOSMEM: s_cbranch_execz
; TOSMEM: s_branch

; TOSMEM: BB{{[0-9]+_[0-9]+}}:
; TOSMEM: s_add_u32 m0, s3, 0x100
; TOSMEM-NEXT: s_buffer_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, m0 ; 8-byte Folded Reload

; GCN-NOT: v_readlane_b32 m0
; GCN-NOT: s_buffer_store_dword m0
; GCN-NOT: s_buffer_load_dword m0
define amdgpu_kernel void @m0_unavailable_spill(i32 %m0.arg) #0 {
main_body:
  %m0 = call i32 asm sideeffect "; def $0, 1", "={M0}"() #0
  %tmp = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %m0.arg)
  call void asm sideeffect "; clobber $0", "~{M0}"() #0
  %cmp = fcmp ueq float 0.000000e+00, %tmp
   br i1 %cmp, label %if, label %else

if:                                               ; preds = %main_body
  store volatile i32 8, i32 addrspace(1)* undef
  br label %endif

else:                                             ; preds = %main_body
  store volatile i32 11, i32 addrspace(1)* undef
  br label %endif

endif:
  ret void
}

; GCN-LABEL: {{^}}restore_m0_lds:
; TOSMEM: s_load_dwordx2 [[REG:s\[[0-9]+:[0-9]+\]]]
; TOSMEM: s_cmp_eq_u32
; FIXME: RegScavenger::isRegUsed() always returns true if m0 is reserved, so we have to save and restore it
; FIXME-TOSMEM-NOT: m0
; TOSMEM: s_add_u32 m0, s3, 0x100
; TOSMEM: s_buffer_store_dwordx2 [[REG]], s[88:91], m0 ; 8-byte Folded Spill
; FIXME-TOSMEM-NOT: m0
; TOSMEM: s_add_u32 m0, s3, 0x300
; TOSMEM: s_buffer_store_dword s{{[0-9]+}}, s[88:91], m0 ; 4-byte Folded Spill
; FIXME-TOSMEM-NOT: m0
; TOSMEM: s_cbranch_scc1

; TOSMEM: s_mov_b32 m0, -1

; TOSMEM: s_mov_b32 s0, m0
; TOSMEM: s_add_u32 m0, s3, 0x100
; TOSMEM: s_buffer_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[88:91], m0 ; 8-byte Folded Reload
; TOSMEM: s_mov_b32 m0, s0
; TOSMEM: s_waitcnt lgkmcnt(0)

; TOSMEM: ds_write_b64

; FIXME-TOSMEM-NOT: m0
; TOSMEM: s_add_u32 m0, s3, 0x300
; TOSMEM: s_buffer_load_dword s0, s[88:91], m0 ; 4-byte Folded Reload
; FIXME-TOSMEM-NOT: m0
; TOSMEM: s_waitcnt lgkmcnt(0)
; TOSMEM-NOT: m0
; TOSMEM: s_mov_b32 m0, s0
; TOSMEM: ; use m0

; TOSMEM: s_dcache_wb
; TOSMEM: s_endpgm
define amdgpu_kernel void @restore_m0_lds(i32 %arg) {
  %m0 = call i32 asm sideeffect "s_mov_b32 m0, 0", "={M0}"() #0
  %sval = load volatile i64, i64 addrspace(2)* undef
  %cmp = icmp eq i32 %arg, 0
  br i1 %cmp, label %ret, label %bb

bb:
  store volatile i64 %sval, i64 addrspace(3)* undef
  call void asm sideeffect "; use $0", "{M0}"(i32 %m0) #0
  br label %ret

ret:
  ret void
}

declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #1
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #0
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
