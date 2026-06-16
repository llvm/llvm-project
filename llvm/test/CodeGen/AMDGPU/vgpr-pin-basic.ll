; Check the void pin marker selects to SI_VGPR_PIN, the pass consumes it, and
; codegen reaches s_endpgm -- on both SDAG and GISel (-global-isel-abort=1
; makes a fallback a hard error so BEFORE-PIN can't be satisfied by SDAG).
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O2 -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=BEFORE-PIN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O2 -verify-machineinstrs -stop-after=si-pin-vgpr < %s | FileCheck %s --check-prefix=AFTER-PIN
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O2 -verify-machineinstrs -filetype=asm < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O2 -verify-machineinstrs -stop-after=instruction-select < %s | FileCheck %s --check-prefix=BEFORE-PIN
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O2 -verify-machineinstrs -stop-after=si-pin-vgpr < %s | FileCheck %s --check-prefix=AFTER-PIN
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O2 -verify-machineinstrs -filetype=asm < %s | FileCheck %s --check-prefix=ASM
;
; -O0: SIPinVGPR doesn't run, so expandPostRAPseudo must erase the marker --
; otherwise the meta-pseudo is silently dropped by the AsmPrinter.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O0 -verify-machineinstrs -filetype=asm < %s | FileCheck %s --check-prefix=ASM-O0
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O0 -verify-machineinstrs -filetype=asm < %s | FileCheck %s --check-prefix=ASM-O0

; BEFORE-PIN-LABEL: name:{{ +}}vgpr_pin_i32
; BEFORE-PIN:       SI_VGPR_PIN

; AFTER-PIN-LABEL: name:{{ +}}vgpr_pin_i32
; AFTER-PIN-NOT:   SI_VGPR_PIN
; AFTER-PIN:       S_ENDPGM 0

; ASM-LABEL: vgpr_pin_i32:
; ASM:       s_endpgm

; -O0: a "; meta instruction" comment would mean SI_VGPR_PIN reached the
; AsmPrinter unconsumed; its absence proves expandPostRAPseudo erased it.
; ASM-O0-LABEL:  vgpr_pin_i32:
; ASM-O0-NOT:    ; meta instruction
; ASM-O0:        s_endpgm
define amdgpu_kernel void @vgpr_pin_i32(ptr addrspace(1) %out) {
entry:
  %in = call i32 asm sideeffect "; produce in", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.i32(i32 %in)
  %add = add i32 %in, 1
  store i32 %add, ptr addrspace(1) %out, align 4
  ret void
}

; BEFORE-PIN-LABEL: name:{{ +}}vgpr_pin_v4i32
; BEFORE-PIN:       SI_VGPR_PIN

; AFTER-PIN-LABEL: name:{{ +}}vgpr_pin_v4i32
; AFTER-PIN-NOT:   SI_VGPR_PIN
; AFTER-PIN:       S_ENDPGM 0

; ASM-LABEL: vgpr_pin_v4i32:
; ASM:       s_endpgm
define amdgpu_kernel void @vgpr_pin_v4i32(ptr addrspace(1) %out) {
entry:
  %in = call <4 x i32> asm sideeffect "; produce in", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v4i32(<4 x i32> %in)
  store <4 x i32> %in, ptr addrspace(1) %out, align 16
  ret void
}

; BEFORE-PIN-LABEL: name:{{ +}}vgpr_pin_v32f32
; BEFORE-PIN:       SI_VGPR_PIN

; AFTER-PIN-LABEL: name:{{ +}}vgpr_pin_v32f32
; AFTER-PIN-NOT:   SI_VGPR_PIN
; AFTER-PIN:       S_ENDPGM 0

; ASM-LABEL: vgpr_pin_v32f32:
; ASM:       s_endpgm
define amdgpu_kernel void @vgpr_pin_v32f32(ptr addrspace(1) %out) {
entry:
  %in = call <32 x float> asm sideeffect "; produce in", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v32f32(<32 x float> %in)
  store <32 x float> %in, ptr addrspace(1) %out, align 128
  ret void
}

declare void @llvm.amdgcn.internal.vgpr.pin.i32(i32)
declare void @llvm.amdgcn.internal.vgpr.pin.v4i32(<4 x i32>)
declare void @llvm.amdgcn.internal.vgpr.pin.v32f32(<32 x float>)
