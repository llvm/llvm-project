; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown < %s 2>&1 | FileCheck %s
; CHECK: unable to translate instruction: call (in function: kernel)

; Interposable aliases are not yet supported.
@bar_alias = weak alias void (), ptr addrspace(4) @bar

; Interposable functions are not yet supported for aliasing resolution.
define weak spir_func void @bar() addrspace(4) {
entry:
  ret void
}

define spir_kernel void @kernel() addrspace(4) {
entry:
  call spir_func addrspace(4) void @bar_alias()
  ret void
}
