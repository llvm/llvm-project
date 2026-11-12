; RUN: llc -mtriple x86_64-pc-linux %s -o - | FileCheck %s

module asm ".pushsection foo"
module asm ".popsection"

; CHECK: .section	foo,"",@progbits
; CHECK: .text
