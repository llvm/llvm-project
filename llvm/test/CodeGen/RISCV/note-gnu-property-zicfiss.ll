; RUN: llc --mtriple=riscv32 --filetype=obj -o - %s | llvm-readelf -n - | FileCheck --check-prefixes=READELF %s
; RUN: llc --mtriple=riscv64 --filetype=obj -o - %s | llvm-readelf -n - | FileCheck --check-prefixes=READELF %s
; RUN: llc --mtriple=riscv32 -o - %s | FileCheck --check-prefixes=ASM,ASM32 %s
; RUN: llc --mtriple=riscv64 -o - %s | FileCheck --check-prefixes=ASM,ASM64 %s

; READELF: Properties: RISC-V feature: ZICFISS

; ASM:                .section        ".note.GNU-stack","",@progbits
; ASM-NEXT:           .section        .note.gnu.property,"a",@note
; ASM32-NEXT:         .p2align        2, 0x0
; ASM64-NEXT:         .p2align        3, 0x0
; ASM-NEXT:           .word   4
; ASM32-NEXT:         .word   12
; ASM64-NEXT:         .word   16
; ASM-NEXT:           .word   5
; ASM-NEXT:           .asciz  "GNU"
; ASM-NEXT:           .word   3221225472
; ASM-NEXT:           .word   4
; ASM-NEXT:           .word   2
; ASM32-NEXT:         .p2align        2, 0x0
; ASM64-NEXT:         .p2align        3, 0x0

define i32 @f() "hw-shadow-stack" {
entry:
  ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"cf-protection-return", i32 1}
