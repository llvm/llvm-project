; NOTE: Do not autogenerate. Object emission invokes the inline assembler; null emission does not. Check warning presence and absence.
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.o < %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -filetype=obj -o %t.o -amdgpu-asm-warn-async-barrier-replay=false < %s 2>&1 | count 0
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-xnack=0 -filetype=obj -o %t.o < %s 2>&1 | count 0
; RUN: llc -mtriple=amdgpu12.51-amd-amdhsa -filetype=obj -o %t.o < %s 2>&1 | FileCheck %s --check-prefix=WARN --implicit-check-not=warning:

define amdgpu_kernel void @unprotected_inline_asm() {
  call void asm sideeffect "ds_atomic_async_barrier_arrive_b64 v0", "~{v0},~{memory}"()
; WARN: warning: no recognized replay workaround after 'ds_atomic_async_barrier_arrive_b64' on this GPU
  ret void
}

define amdgpu_kernel void @protected_inline_asm() {
  call void asm sideeffect "ds_atomic_async_barrier_arrive_b64 v0\0As_mov_b32 s7, exec_lo\0As_mov_b32 exec_lo, 0\0Aglobal_load_async_to_lds_b8 v0, v[0:1], off\0As_mov_b32 exec_lo, s7", "~{v0},~{v1},~{s7},~{memory}"()
  ret void
}

; A later inline-asm block cannot protect an arrival in an earlier block.
define amdgpu_kernel void @split_inline_asm() {
  call void asm sideeffect "ds_atomic_async_barrier_arrive_b64 v0", "~{v0},~{memory}"()
; WARN: warning: no recognized replay workaround after 'ds_atomic_async_barrier_arrive_b64' on this GPU
  call void asm sideeffect "s_mov_b32 s7, exec_lo\0As_mov_b32 exec_lo, 0\0Aglobal_load_async_to_lds_b8 v0, v[0:1], off\0As_mov_b32 exec_lo, s7", "~{v0},~{v1},~{s7},~{memory}"()
  ret void
}
