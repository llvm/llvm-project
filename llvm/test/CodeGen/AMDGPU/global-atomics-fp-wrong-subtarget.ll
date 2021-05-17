; RUN: llc -march=amdgcn -mcpu=gfx803 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx803 -filetype=obj < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx803 -d - | FileCheck -check-prefix=DISASSEMBLY-VI %s

; Make sure we can encode and don't fail on functions which have
; instructions not actually supported by the subtarget.
; FIXME: This will still fail for gfx6/7 and gfx10 subtargets.

; DISASSEMBLY-VI: .long 0xdd348000                                           // {{[0-9]+}}: DD348000
; DISASSEMBLY-VI-NEXT: v_cndmask_b32_e32 v0, v0, v0, vcc                     // {{[0-9]+}}: 00000100

define amdgpu_kernel void @global_atomic_fadd_noret_f32_wrong_subtarget(float addrspace(1)* %ptr) #0 {
; GCN-LABEL: global_atomic_fadd_noret_f32_wrong_subtarget:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x24
; GCN-NEXT:    v_mov_b32_e32 v0, 0
; GCN-NEXT:    v_mov_b32_e32 v1, 4.0
; GCN-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; GCN-NEXT:    global_atomic_add_f32 v0, v1, s[0:1]
; GCN-NEXT:    s_waitcnt vmcnt(0)
; GCN-NEXT:    buffer_wbinvl1_vol
; GCN-NEXT:    s_endpgm
  %result = atomicrmw fadd float addrspace(1)* %ptr, float 4.0 syncscope("agent") seq_cst
  ret void
}

attributes #0 = { "denormal-fp-math-f32"="preserve-sign,preserve-sign" "target-features"="+atomic-fadd-insts" "amdgpu-unsafe-fp-atomics"="true" }
