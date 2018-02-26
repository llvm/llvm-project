; RUN: llc -mtriple=mips-unknown-linux -filetype=obj %s -o - | \
; RUN:   llvm-readobj -mips-abi-flags | \
; RUN:   FileCheck --check-prefix=ASE-MICROMIPS %s

define void @_Z3foov() #0 {
entry:
  ret void
}
attributes #0 = { "micromips" }

; ASE-MICROMIPS: microMIPS (0x800)
