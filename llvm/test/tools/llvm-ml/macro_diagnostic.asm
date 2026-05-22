; RUN: not llvm-ml -filetype=s %s /Fo - 2>&1 | FileCheck %s

.code

test_macro macro
  invalid_instruction_here
endm

; CHECK: macro_diagnostic.asm:6:1: error: invalid instruction mnemonic 'invalid_instruction_here'
; CHECK: macro_diagnostic.asm:11:1: note: while in macro instantiation
test_macro
end
