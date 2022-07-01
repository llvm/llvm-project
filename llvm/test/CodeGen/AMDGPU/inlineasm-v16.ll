; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s
; RUN: not llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s 2>&1 | FileCheck -enable-var-scope -check-prefixes=INVALID %s
; RUN: not llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s 2>&1 | FileCheck -enable-var-scope -check-prefixes=INVALID %s

; GCN-LABEL: {{^}}s_input_output_v8f16
; GCN: s_mov_b32 s[0:3], -1
; GCN: ; use s[0:3]
; INVALID: error: couldn't allocate output register for constraint 's'
; INVALID: error: couldn't allocate input reg for constraint 's'
define amdgpu_kernel void @s_input_output_v8f16() {
  %v = tail call <8 x half> asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(<8 x half> %v)
  ret void
}

; GCN-LABEL: {{^}}s_input_output_v8i16
; GCN: s_mov_b32 s[0:3], -1
; GCN: ; use s[0:3]
; INVALID: error: couldn't allocate output register for constraint 's'
; INVALID: error: couldn't allocate input reg for constraint 's'
define amdgpu_kernel void @s_input_output_v8i16() {
  %v = tail call <8 x i16> asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(<8 x i16> %v)
  ret void
}

; GCN-LABEL: {{^}}v_input_output_v8f16
; GCN: v_mov_b32 v[0:3], -1
; GCN: ; use v[0:3]
; INVALID: error: couldn't allocate output register for constraint 'v'
; INVALID: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_output_v8f16() {
  %v = tail call <8 x half> asm sideeffect "v_mov_b32 $0, -1", "=v"()
  tail call void asm sideeffect "; use $0", "v"(<8 x half> %v)
  ret void
}

; GCN-LABEL: {{^}}v_input_output_v8i16
; GCN: v_mov_b32 v[0:3], -1
; GCN: ; use v[0:3]
; INVALID: error: couldn't allocate output register for constraint 'v'
; INVALID: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_output_v8i16() {
  %v = tail call <8 x i16> asm sideeffect "v_mov_b32 $0, -1", "=v"()
  tail call void asm sideeffect "; use $0", "v"(<8 x i16> %v)
  ret void
}

; GCN-LABEL: {{^}}s_input_output_v16f16
; GCN: s_mov_b32 s[0:7], -1
; GCN: ; use s[0:7]
; INVALID: error: couldn't allocate output register for constraint 's'
; INVALID: error: couldn't allocate input reg for constraint 's'
define amdgpu_kernel void @s_input_output_v16f16() {
  %v = tail call <16 x half> asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(<16 x half> %v)
  ret void
}

; GCN-LABEL: {{^}}s_input_output_v16i16
; GCN: s_mov_b32 s[0:7], -1
; GCN: ; use s[0:7]
; INVALID: error: couldn't allocate output register for constraint 's'
; INVALID: error: couldn't allocate input reg for constraint 's'
define amdgpu_kernel void @s_input_output_v16i16() {
  %v = tail call <16 x i16> asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(<16 x i16> %v)
  ret void
}

; GCN-LABEL: {{^}}v_input_output_v16f16
; GCN: v_mov_b32 v[0:7], -1
; GCN: ; use v[0:7]
; INVALID: error: couldn't allocate output register for constraint 'v'
; INVALID: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_output_v16f16() {
  %v = tail call <16 x half> asm sideeffect "v_mov_b32 $0, -1", "=v"()
  tail call void asm sideeffect "; use $0", "v"(<16 x half> %v)
  ret void
}

; GCN-LABEL: {{^}}v_input_output_v16i16
; GCN: v_mov_b32 v[0:7], -1
; GCN: ; use v[0:7]
; INVALID: error: couldn't allocate output register for constraint 'v'
; INVALID: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_output_v16i16() {
  %v = tail call <16 x i16> asm sideeffect "v_mov_b32 $0, -1", "=v"()
  tail call void asm sideeffect "; use $0", "v"(<16 x i16> %v)
  ret void
}

attributes #0 = { nounwind }
