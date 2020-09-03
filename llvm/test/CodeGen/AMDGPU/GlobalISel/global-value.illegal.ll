; RUN: not --crash llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -o /dev/null %s 2>&1 | FileCheck -check-prefix=ERR %s

; FIXME: Should produce context error for each one
; ERR: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(p5) = G_GLOBAL_VALUE @external_private (in function: fn_external_private)

@external_private = external addrspace(5) global i32, align 4
@internal_private = internal addrspace(5) global i32 undef, align 4

define i32 addrspace(5)* @fn_external_private() {
  ret i32 addrspace(5)* @external_private
}

define i32 addrspace(5)* @fn_internal_private() {
  ret i32 addrspace(5)* @internal_private
}
