; RUN: llc --mtriple=riscv32 --filetype=obj -o - %s | llvm-readelf -n - | FileCheck --check-prefixes=READELF %s
; RUN: llc --mtriple=riscv64 --filetype=obj -o - %s | llvm-readelf -n - | FileCheck --check-prefixes=READELF %s
; RUN: llc --mtriple=riscv32 -o - %s | FileCheck --check-prefixes=ASM,ASM32 %s
; RUN: llc --mtriple=riscv64 -o - %s | FileCheck --check-prefixes=ASM,ASM64 %s

; READELF: Properties: RISC-V feature: ZICFILP-unlabeled

; ASM:                .section        ".note.GNU-stack","",@progbits
; ASM-NEXT:           .section        .note.gnu.property,"a",@note
; ASM-NEXT:           .word   4
; ASM-NEXT:           .word   .Ltmp1-.Ltmp0
; ASM-NEXT:           .word   5
; ASM-NEXT:           .asciz  "GNU"
; ASM-NEXT:   .Ltmp0:
; ASM32-NEXT:         .p2align        2, 0x0
; ASM64-NEXT:         .p2align        3, 0x0
; ASM-NEXT:           .word   3221225472
; ASM-NEXT:           .word   4
; ASM-NEXT:           .word   1
; ASM32-NEXT:         .p2align        2, 0x0
; ASM64-NEXT:         .p2align        3, 0x0
; ASM-NEXT:   .Ltmp1:

!llvm.module.flags = !{!0, !1}

!0 = !{i32 8, !"cf-protection-branch", i32 1}
!1 = !{i32 1, !"cf-branch-label-scheme", !"unlabeled"}
