; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null 2>&1 %s | FileCheck -check-prefix=ERR %s

; ERR: warning: inline asm clobber list contains reserved registers: v42
; ERR: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
define amdgpu_kernel void @clobber_occupancy_limited_vgpr() #0 {
entry:
  call void asm sideeffect "; clobber $0", "~{v42}"()
  ret void
}

; ERR: warning: inline asm clobber list contains reserved registers: v[42:43]
; ERR: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
define amdgpu_kernel void @clobber_occupancy_limited_vgpr64() #0 {
entry:
  call void asm sideeffect "; clobber $0", "~{v[42:43]}"()
  ret void
}

; ERR: warning: inline asm clobber list contains reserved registers: m0
; ERR: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
define amdgpu_kernel void @clobber_m0() {
entry:
  call void asm sideeffect "; clobber $0", "~{m0}"()
  ret void
}

; ERR: warning: inline asm clobber list contains reserved registers: exec
; ERR: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
define amdgpu_kernel void @clobber_exec() {
entry:
  call void asm sideeffect "; clobber $0", "~{exec}"()
  ret void
}

; ERR: warning: inline asm clobber list contains reserved registers: exec_lo
; ERR: note: Reserved registers on the clobber list may not be preserved across the asm statement, and clobbering them may lead to undefined behaviour.
define amdgpu_kernel void @clobber_exec_lo() {
entry:
  call void asm sideeffect "; clobber $0", "~{exec_lo}"()
  ret void
}

; FIXME: This should warn too
; ERR-NOT: warning: inline asm clobber list contains reserved registers
define amdgpu_kernel void @def_exec(ptr addrspace(1) %ptr) {
entry:
  %exec = call i64 asm sideeffect "; def $0", "={exec}"()
  store i64 %exec, ptr addrspace(1) %ptr
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="10,10" }
