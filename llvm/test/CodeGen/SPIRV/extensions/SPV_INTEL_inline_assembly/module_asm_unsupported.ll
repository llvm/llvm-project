; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; SPIR-V has no binary analog for module-level inline assembly.
; The backend must reject it with a clean diagnostic
; rather than reach the unimplemented MC asm parser path.

; CHECK: error: SPIR-V does not support module-level inline assembly

module asm "foo"

define void @main() {
entry:
  ret void
}
