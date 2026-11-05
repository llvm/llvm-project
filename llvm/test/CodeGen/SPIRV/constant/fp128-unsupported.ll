; Test that fp128 and ppc_fp128, which are not valid standard SPIR-V types,
; produce an error instead of being silently miscompiled.

; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: fp128 is not supported in SPIR-V

define fp128 @getConstantFP128() {
  ret fp128 0xL00000000000000004001000000000000
}
