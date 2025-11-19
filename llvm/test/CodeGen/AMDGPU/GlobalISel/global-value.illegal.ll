; RUN: not llc -global-isel -global-isel-abort=1 -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s

; FIXME: Should produce context error for each one
; ERR: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(p5) = G_GLOBAL_VALUE @external_private (in function: fn_external_private)

@external_private = external addrspace(5) global i32, align 4
@internal_private = internal addrspace(5) global i32 poison, align 4

define ptr addrspace(5) @fn_external_private() {
  ret ptr addrspace(5) @external_private
}

define ptr addrspace(5) @fn_internal_private() {
  ret ptr addrspace(5) @internal_private
}
