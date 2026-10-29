; RUN: llvm-split -enable-call-graph-split-module=true -j2 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; Test module-level inline asm.

module asm ".globl __split_asm_marker"
module asm "__split_asm_marker:"
module asm ".long 42"

; CHECK0: module asm
; CHECK0:     ".globl __split_asm_marker"
; CHECK1-NOT: module asm
; CHECK1-NOT:     ".globl __split_asm_marker"

define void @foo() {
entry:
  ret void
}

define void @bar() {
entry:
  ret void
}
