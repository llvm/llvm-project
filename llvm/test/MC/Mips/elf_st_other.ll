; This tests value of ELF st_other field for function symbol table entries.
; For microMIPS value should be equal to STO_MIPS_MICROMIPS.

; RUN: llc -mtriple mipsel-unknown-linux -mcpu=mips32r2 -mattr=+micromips -print-hack-directives %s -o - | FileCheck %s

define i32 @main() nounwind {
entry:
  ret i32 0
}

; CHECK:     .mips_hack_stocg main, 128
