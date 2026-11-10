; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 -stop-after=finalize-isel < %s | FileCheck --check-prefix=MIR %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 -stop-after=finalize-isel < %s | FileCheck --check-prefix=MIR %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 < %s | FileCheck --check-prefix=ASM %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 < %s | FileCheck --check-prefix=ASM %s

; With true16 a 16-bit return value comes back in the lo16 subregister of a
; VGPR that csr_amdgpu_si_gfx still claims is callee saved. The callee drops
; the whole 32-bit register from its saved set and clobbers all 32 bits, so the
; call has to mark the enclosing register as clobbered explicitly -- otherwise
; a value could stay live in the hi16 half across the call. Check that both the
; SelectionDAG and the GlobalISel call lowering do so.

declare amdgpu_gfx { [40 x float], half } @external_return_40xfloat_half(half)

; MIR-LABEL: name: call_return_40xfloat_half
; MIR: SI_CALL {{.*}}csr_amdgpu_si_gfx{{.*}}implicit-def $vgpr40_lo16{{.*}}implicit-def $vgpr40{{$}}
define amdgpu_gfx half @call_return_40xfloat_half(half %a, half %b) #0 {
entry:
  %live = fadd half %a, 1.0
  %r = call amdgpu_gfx { [40 x float], half } @external_return_40xfloat_half(half %b)
  %h = extractvalue { [40 x float], half } %r, 1
  %s = fadd half %live, %h
  ret half %s
}

; The callee side of the same ABI rule: v40 holds the returned half in its lo16
; half, so it must not be spilled and restored -- the restore in the epilogue
; would clobber the returned value.

; ASM-LABEL: {{^}}return_40xfloat_half:
; ASM-NOT:   scratch_
; ASM:       v_add_f16_e32 v40.l, 1.0, v0.l
; ASM-NEXT:  s_setpc_b64
define amdgpu_gfx { [40 x float], half } @return_40xfloat_half(half %x) #0 {
entry:
  %y = fadd half %x, 1.0
  %r = insertvalue { [40 x float], half } poison, half %y, 1
  ret { [40 x float], half } %r
}

; Same as above, but the function also clobbers all 32 bits of v40 through
; inline asm. v40 must still be dropped from the saved set entirely, so the
; clobber is left undone rather than repaired by a reload.

; ASM-LABEL: return_40xfloat_half_clobber_v40:
; ASM-NOT:   scratch_
; ASM:       v_mov_b16_e32 v40.l,
; ASM-NEXT:  s_setpc_b64
define amdgpu_gfx { [40 x float], half } @return_40xfloat_half_clobber_v40(half %x) #0 {
entry:
  call void asm sideeffect "v_mov_b32 v40, 0", "~{v40}"()
  %y = fadd half %x, 1.0
  %r = insertvalue { [40 x float], half } poison, half %y, 1
  ret { [40 x float], half } %r
}

attributes #0 = { nounwind "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,1" }
