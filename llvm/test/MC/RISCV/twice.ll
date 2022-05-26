; Check for state persistence bugs in the RISC-V MC backend
; This should neither fail (in the comparison that the second object
; is bit-identical to the first) nor crash. Either failure would most
; likely indicate some state that is not properly reset in the
; appropriate ::reset method.
; RUN: llc -compile-twice -filetype=obj -mtriple=riscv64 %s -o - \
; RUN:     | llvm-objdump --section-headers - \
; RUN:     | FileCheck %s

; CHECK:      Sections:
; CHECK-NEXT: Idx Name              Size     VMA              Type
; CHECK-NEXT:  0
; CHECK-NEXT:  1 .strtab
; CHECK-NEXT:  2 .text
; CHECK-NEXT:  3 .note.GNU-stack
; CHECK-NEXT:  4 .riscv.attributes
; CHECK-NEXT:  5 .symtab
