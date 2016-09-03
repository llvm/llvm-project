; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=1 -march=amdgcn -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck -check-prefix=TOVGPR -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=1 -march=amdgcn -mcpu=tonga -mattr=+vgpr-spilling  -verify-machineinstrs < %s | FileCheck -check-prefix=TOVGPR -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=0 -march=amdgcn -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck -check-prefix=TOVMEM -check-prefix=GCN %s
; RUN: llc -O0 -amdgpu-spill-sgpr-to-vgpr=0 -march=amdgcn -mattr=+vgpr-spilling -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=TOVMEM -check-prefix=GCN %s

; XXX - Why does it like to use vcc?

; GCN-LABEL: {{^}}spill_m0:
; GCN: s_cmp_lg_i32

; TOVGPR: s_mov_b32 vcc_hi, m0
; TOVGPR: v_writelane_b32 [[SPILL_VREG:v[0-9]+]], vcc_hi, 0

; TOVMEM: v_mov_b32_e32 [[SPILL_VREG:v[0-9]+]], m0
; TOVMEM: buffer_store_dword [[SPILL_VREG]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} ; 4-byte Folded Spill
; TOVMEM: s_waitcnt vmcnt(0)
; GCN: s_cbranch_scc1 [[ENDIF:BB[0-9]+_[0-9]+]]

; GCN: [[ENDIF]]:
; TOVGPR: v_readlane_b32 vcc_hi, [[SPILL_VREG]], 0
; TOVGPR: s_mov_b32 m0, vcc_hi

; TOVMEM: buffer_load_dword [[RELOAD_VREG:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} ; 4-byte Folded Reload
; TOVMEM: s_waitcnt vmcnt(0)
; TOVMEM: v_readfirstlane_b32 vcc_hi, [[RELOAD_VREG]]
; TOVMEM: s_mov_b32 m0, vcc_hi

; GCN: s_add_i32 m0, m0, 1
define void @spill_m0(i32 %cond, i32 addrspace(1)* %out) #0 {
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

; GCN-LABEL: {{^}}spill_m0_lds:
; GCN-NOT: v_readlane_b32 m0
define amdgpu_ps void @spill_m0_lds(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg) #0 {
main_body:
  %4 = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %3)
  %cmp = fcmp ueq float 0.0, %4
  br i1 %cmp, label %if, label %else

if:
  %lds_ptr = getelementptr [64 x float], [64 x float] addrspace(3)* @lds, i32 0, i32 0
  %lds_data = load float, float addrspace(3)* %lds_ptr
  br label %endif

else:
  %interp = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %3)
  br label %endif

endif:
  %export = phi float [%lds_data, %if], [%interp, %else]
  %5 = call i32 @llvm.SI.packf16(float %export, float %export)
  %6 = bitcast i32 %5 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %6, float %6, float %6, float %6)
  ret void
}

declare float @llvm.SI.fs.constant(i32, i32, i32) readnone

declare i32 @llvm.SI.packf16(float, float) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind }
