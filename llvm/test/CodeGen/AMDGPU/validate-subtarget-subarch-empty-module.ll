; Make sure we still diagose invalid cpu for the subarch even if there
; are no functions in the module.  This differs from
; validate-subtarget-subarch.ll in that a subtarget is never
; constructed.

; RUN: not llc -mtriple=amdgpu9 -mcpu=gfx1030 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=amdgpu12345 -mcpu=gfx600 -filetype=null %s 2>&1 | FileCheck -check-prefix=UNKNOWN-SUBARCH %s

; ERR: LLVM ERROR: invalid cpu 'gfx1030' for subarch amdgpu9
; UNKNOWN-SUBARCH: LLVM ERROR: unknown subarch amdgpu12345
