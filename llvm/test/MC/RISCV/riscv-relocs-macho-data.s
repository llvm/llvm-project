; RUN: llvm-mc -triple riscv32-apple-macho -filetype=obj %s -o %t.o
; RUN: llvm-objdump -dr --section __data --full-contents %t.o | FileCheck %s --check-prefix=CHECK
; RUN: llvm-otool -Vtr %t.o | FileCheck %s --check-prefix=OTOOL

; Data section relocations
        .data
; CHECK:  0000 2a000000 0c000000 00000000 04000000  *...............
; CHECK-NEXT:  0010 00000000 02000000                    ........

; Plain integer, no relocation needed.
        .global _a
; CHECK-LABEL: 00000000 <ltmp1>:
; CHECK-NEXT:        0: 002a            <unknown>
; CHECK-NEXT:        2: 0000            <unknown>
_a:
        .word 42

; Plain integer, no relocation needed.
        .global _b
; CHECK-LABEL: 00000004 <_b>:
; CHECK-NEXT:        4: 000c            <unknown>
; CHECK-NEXT:        6: 0000            <unknown>
_b:
        .word 12

; Pointer to symbol, require relocation. The 2 chunks of 2 bytes
; each at PC=8 and PC=a are filled with 0 as a placeholder.
        .global _ref
; CHECK-LABEL: 00000008 <_ref>:
; CHECK-NEXT:        8: 0000            <unknown>
; CHECK-NEXT:                   00000008:  RISCV_RELOC_UNSIGNED _b
; CHECK-NEXT:        a: 0000            <unknown>
_ref:
        .word _b

; Same as the previous example of pointer to symnbol, but with
; offset. The placeholder data is storing the offset (00000004).
        .global _ref_offset
; CHECK-LABEL: 0000000c <_ref_offset>:
; CHECK-NEXT:        c: 0004            <unknown>
; CHECK-NEXT:                   0000000c:  RISCV_RELOC_UNSIGNED _b
; CHECK-NEXT:        e: 0000            <unknown>
_ref_offset:
        .word _b + 4

        .global _sub
; Difference of addresses to symbols. The linker will take care of
; replacing the 4 bytes 00000000 with the difference of the address
; of _ref minus the addresss of _elsewhere.
_sub:
; CHECK-LABEL: 00000010 <_sub>:
; CHECK-NEXT:       10: 0000            <unknown>
; CHECK-NEXT:                   00000010:  RISCV_RELOC_SUBTRACTOR       _elsewhere
; CHECK-NEXT:                   00000010:  RISCV_RELOC_UNSIGNED _ref
; CHECK-NEXT:       12: 0000            <unknown>
        .word _ref - _elsewhere


; Same as before, but an additional offset is stored in the
; placeholder 4 bytes to be used as the addend in the expression.
        .global _sub_add
_sub_add:
; CHECK-LABEL: 00000014 <_sub_add>:
; CHECK-NEXT:       14: 0002            <unknown>
; CHECK-NEXT:                   00000014:  RISCV_RELOC_SUBTRACTOR       _elsewhere
; CHECK-NEXT:                   00000014:  RISCV_RELOC_UNSIGNED _ref
; CHECK-NEXT:       16: 0000            <unknown>
.word _ref - _elsewhere + 2
; CHECK-NOT: {{.}}

; OTOOL-LABEL: Relocation information (__DATA,__data) 6 entries
; OTOOL-NEXT:  address  pcrel length extern type    scattered symbolnum/value
; OTOOL-NEXT:  00000014 False long   True   1       False     _elsewhere
; OTOOL-NEXT:  00000014 False long   True   0       False     _ref
; OTOOL-NEXT:  00000010 False long   True   1       False     _elsewhere
; OTOOL-NEXT:  00000010 False long   True   0       False     _ref
; OTOOL-NEXT:  0000000c False long   True   0       False     _b
; OTOOL-NEXT:  00000008 False long   True   0       False     _b
; OTOOL-NEXT:  Contents of (__TEXT,__text) section
; OTOOL-NOT: {{.}}
