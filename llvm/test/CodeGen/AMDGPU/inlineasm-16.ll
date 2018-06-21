; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: not llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=SICI %s
; RUN: not llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=SICI %s

; GCN-LABEL: {{^}}s_input_output_i16:
; SICI: error: couldn't allocate output register for constraint 's'
; SICI: error: couldn't allocate input reg for constraint 's'
define amdgpu_kernel void @s_input_output_i16() #0 {
  %v = tail call i16 asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(i16 %v) #0
  ret void
}

; GCN-LABEL: {{^}}v_input_output_i16:
; SICI: error: couldn't allocate output register for constraint 'v'
; SICI: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_output_i16() #0 {
  %v = tail call i16 asm sideeffect "v_mov_b32 $0, -1", "=v"() #0
  tail call void asm sideeffect "; use $0", "v"(i16 %v)
  ret void
}

; GCN-LABEL: {{^}}s_input_output_f16:
; SICI: error: couldn't allocate output register for constraint 's'
; SICI: error: couldn't allocate input reg for constraint 's'
define amdgpu_kernel void @s_input_output_f16() #0 {
  %v = tail call half asm sideeffect "s_mov_b32 $0, -1", "=s"() #0
  tail call void asm sideeffect "; use $0", "s"(half %v)
  ret void
}

; GCN-LABEL: {{^}}v_input_output_f16:
; SICI: error: couldn't allocate output register for constraint 'v'
; SICI: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_output_f16() #0 {
  %v = tail call half asm sideeffect "v_mov_b32 $0, -1", "=v"() #0
  tail call void asm sideeffect "; use $0", "v"(half %v)
  ret void
}

attributes #0 = { nounwind }
