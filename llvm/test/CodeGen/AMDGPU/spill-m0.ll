; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=1 -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=TOVGPR -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=1 -march=amdgcn -mcpu=tonga  -verify-machineinstrs < %s | FileCheck -check-prefix=TOVGPR -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=0 -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=TOVMEM -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=0 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=TOVMEM -check-prefix=GCN %s

; XXX - Why does it like to use vcc?

; GCN-LABEL: {{^}}spill_m0:

; GCN-DAG: s_cmp_lg_u32

; TOVGPR-DAG: s_mov_b32 [[M0_COPY:s[0-9]+]], m0
; TOVGPR: v_writelane_b32 [[SPILL_VREG:v[0-9]+]], [[M0_COPY]], 2

; TOVMEM-DAG: s_mov_b32 [[M0_COPY:s[0-9]+]], m0
; TOVMEM-DAG: v_mov_b32_e32 [[SPILL_VREG:v[0-9]+]], [[M0_COPY]]
; TOVMEM: buffer_store_dword [[SPILL_VREG]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:12 ; 4-byte Folded Spill

; GCN: s_cbranch_scc1 [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: [[ENDIF]]:
; TOVGPR: v_readlane_b32 [[M0_RESTORE:s[0-9]+]], [[SPILL_VREG]], 2
; TOVGPR: s_mov_b32 m0, [[M0_RESTORE]]

; TOVMEM: buffer_load_dword [[RELOAD_VREG:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:12 ; 4-byte Folded Reload
; TOVMEM: s_waitcnt vmcnt(0)
; TOVMEM: v_readfirstlane_b32 [[M0_RESTORE:s[0-9]+]], [[RELOAD_VREG]]
; TOVMEM: s_mov_b32 m0, [[M0_RESTORE]]

; GCN: s_add_i32 s{{[0-9]+}}, m0, 1
define amdgpu_kernel void @spill_m0(i32 %cond, i32 addrspace(1)* %out) #0 {
entry:
  %m0 = call i32 asm sideeffect "s_mov_b32 m0, 0", "={m0}"() #0
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %endif

if:
  call void asm sideeffect "v_nop", ""() #0
  br label %endif

endif:
  %foo = call i32 asm sideeffect "s_add_i32 $0, $1, 1", "=s,{m0}"(i32 %m0) #0
  store i32 %foo, i32 addrspace(1)* %out
  ret void
}

@lds = internal addrspace(3) global [64 x float] undef

; m0 is killed, so it isn't necessary during the entry block spill to preserve it
; GCN-LABEL: {{^}}spill_kill_m0_lds:
; GCN: s_mov_b32 m0, s6
; GCN: v_interp_mov_f32

; GCN-NOT: v_readlane_b32 m0
; GCN-NOT: s_buffer_store_dword m0
; GCN-NOT: s_buffer_load_dword m0
define amdgpu_ps void @spill_kill_m0_lds(<16 x i8> addrspace(4)* inreg %arg, <16 x i8> addrspace(4)* inreg %arg1, <32 x i8> addrspace(4)* inreg %arg2, i32 inreg %m0) #0 {
main_body:
  %tmp = call float @llvm.amdgcn.interp.mov(i32 2, i32 0, i32 0, i32 %m0)
  %cmp = fcmp ueq float 0.000000e+00, %tmp
  br i1 %cmp, label %if, label %else

if:                                               ; preds = %main_body
  %lds_ptr = getelementptr [64 x float], [64 x float] addrspace(3)* @lds, i32 0, i32 0
  %lds_data_ = load float, float addrspace(3)* %lds_ptr
  %lds_data = call float @llvm.amdgcn.wqm.f32(float %lds_data_)
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

declare float @llvm.amdgcn.interp.mov(i32, i32, i32, i32) #1
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #0
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1
declare float @llvm.amdgcn.wqm.f32(float) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
