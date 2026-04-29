; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -O3 -verify-machineinstrs < %s | FileCheck %s

; On gfx90a, an inreg call argument can be allocated to an AV_* register
; class (VGPR-or-AGPR). The waterfall loop in SIInstrInfo's
; emitLoadScalarOpsFromVGPRLoop must not feed an AGPR source directly into
; V_READFIRSTLANE_B32 / V_CMP_EQ_U32_e64; doing so trips the verifier with
; "Operand has incorrect register class" (`$sgprN = V_READFIRSTLANE_B32
; $agpr0`) and aborts in LiveRangeCalc / MachineVerifier. The fix demotes
; the operand into a pure VGPR class via a COPY before reading it in the
; loop.

@G   = global ptr poison
@G.1 = global ptr poison
@G.2 = global ptr addrspace(1) poison
@G.3 = global ptr addrspace(3) poison
@G.4 = global ptr addrspace(5) poison
@G.5 = global ptr addrspace(5) poison

declare amdgpu_cs_chain_preserve void @callee(ptr inreg, ptr,
                                              ptr addrspace(1) inreg,
                                              ptr addrspace(1),
                                              ptr addrspace(3) inreg,
                                              ptr addrspace(3),
                                              ptr addrspace(5) inreg,
                                              ptr addrspace(5))

; The call must waterfall the inreg operands. The source of every emitted
; v_readfirstlane_b32 in this loop must be a VGPR (v#), never an AGPR (a#);
; otherwise we hit "Operand has incorrect register class" on gfx90a.
;
; CHECK-LABEL: chain_preserve_caller:
; CHECK-NOT:  v_readfirstlane_b32 s{{[0-9]+}}, a{{[0-9]+}}
define amdgpu_cs_chain_preserve void @chain_preserve_caller(float inreg %a, float %b) {
  %LGV6 = load ptr addrspace(5), ptr @G.5, align 8
  %LGV5 = load ptr addrspace(5), ptr @G.4, align 8
  %LGV4 = load ptr addrspace(3), ptr @G.3, align 8
  %LGV2 = load ptr addrspace(1), ptr @G.2, align 8
  %LGV1 = load ptr,              ptr @G.1, align 8
  %LGV  = load ptr,              ptr @G,   align 8
  %c = fadd float %a, %b
  store float %c, ptr poison, align 4
  call void @callee(ptr %LGV, ptr %LGV1,
                    ptr addrspace(1) %LGV2, ptr addrspace(1) poison,
                    ptr addrspace(3) poison, ptr addrspace(3) %LGV4,
                    ptr addrspace(5) %LGV5, ptr addrspace(5) %LGV6)
  ret void
}
