; RUN: llc -march=amdgcn -mcpu=gfx803 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx803 -filetype=obj < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx803 -d - | FileCheck -check-prefix=DISASSEMBLY-VI %s

; Make sure we can encode and don't fail on functions which have
; instructions not actually supported by the subtarget.
; FIXME: This will still fail for gfx6/7 and gfx10 subtargets.

; DISASSEMBLY-VI: .long 0xdd348000                                           // {{[0-9A-Z]+}}: DD348000
; DISASSEMBLY-VI-NEXT: v_cndmask_b32_e32 v0, v0, v0, vcc                     // {{[0-9A-Z]+}}: 00000100

define amdgpu_kernel void @global_atomic_fadd_noret_f32_wrong_subtarget(ptr addrspace(1) %ptr) #0 {
; GCN-LABEL: global_atomic_fadd_noret_f32_wrong_subtarget:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_mov_b64 s[2:3], exec
; GCN-NEXT:    v_mbcnt_lo_u32_b32 v0, s2, 0
; GCN-NEXT:    v_mbcnt_hi_u32_b32 v0, s3, v0
; GCN-NEXT:    v_cmp_eq_u32_e32 vcc, 0, v0
; GCN-NEXT:    s_and_saveexec_b64 s[4:5], vcc
; GCN-NEXT:    s_cbranch_execz .LBB0_2
; GCN-NEXT:  ; %bb.1:
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; GCN-NEXT:    s_bcnt1_i32_b64 s2, s[2:3]
; GCN-NEXT:    v_cvt_f32_ubyte0_e32 v1, s2
; GCN-NEXT:    v_mov_b32_e32 v0, 0
; GCN-NEXT:    v_mul_f32_e32 v1, 4.0, v1
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    global_atomic_add_f32 v0, v1, s[0:1]
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    buffer_wbinvl1_vol
; GCN-NEXT:  .LBB0_2:
; GCN-NEXT:    s_endpgm
  %result = atomicrmw fadd ptr addrspace(1) %ptr, float 4.0 syncscope("agent") seq_cst
  ret void
}

attributes #0 = { "denormal-fp-math-f32"="preserve-sign,preserve-sign" "target-features"="+atomic-fadd-no-rtn-insts" "amdgpu-unsafe-fp-atomics"="true" }
