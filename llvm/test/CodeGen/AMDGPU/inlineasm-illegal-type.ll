; RUN: not llc -mtriple=amdgcn -mcpu=bonaire < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=SICI %s
; RUN: not llc -mtriple=amdgcn -mcpu=tonga < %s 2>&1 | FileCheck -check-prefix=GCN %s
; RUN: not llc -mtriple=amdgcn -mcpu=tahiti < %s 2>&1 | FileCheck -check-prefix=GCN -check-prefix=SICI %s

; GCN: error: couldn't allocate output register for constraint 's'
; GCN: error: couldn't allocate input reg for constraint 's'
define amdgpu_kernel void @s_input_output_i8() {
  %v = tail call i8 asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(i8 %v)
  ret void
}

; GCN: error: couldn't allocate output register for constraint 'v'
; GCN: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_output_i8() {
  %v = tail call i8 asm sideeffect "v_mov_b32 $0, -1", "=v"()
  tail call void asm sideeffect "; use $0", "v"(i8 %v)
  ret void
}

; GCN: error: couldn't allocate input reg for constraint 'v'
define amdgpu_kernel void @v_input_empty_struct() {
  call void asm "", "v"({} poison)
  ret void
}

; SICI: error: couldn't allocate output register for constraint 's'
; SICI: error: couldn't allocate input reg for constraint 's'
; VI-NOT: error
define amdgpu_kernel void @s_input_output_v2f16() {
  %v = tail call <2 x half> asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(<2 x half> %v)
  ret void
}

; SICI: error: couldn't allocate output register for constraint 'v'
; SICI: error: couldn't allocate input reg for constraint 'v'
; VI-NOT: error
define amdgpu_kernel void @v_input_output_v2f16() {
  %v = tail call <2 x half> asm sideeffect "v_mov_b32 $0, -1", "=v"()
  tail call void asm sideeffect "; use $0", "v"(<2 x half> %v)
  ret void
}

; SICI: error: couldn't allocate output register for constraint 's'
; SICI: error: couldn't allocate input reg for constraint 's'
; VI-NOT: error
define amdgpu_kernel void @s_input_output_v2i16() {
  %v = tail call <2 x i16> asm sideeffect "s_mov_b32 $0, -1", "=s"()
  tail call void asm sideeffect "; use $0", "s"(<2 x i16> %v)
  ret void
}

; FIXME: Crash in codegen prepare
; define amdgpu_kernel void @s_input_output_i3() {
;   %v = tail call i3 asm sideeffect "s_mov_b32 $0, -1", "=s"()
;   tail call void asm sideeffect "; use $0", "s"(i3 %v)
;   ret void
; }
