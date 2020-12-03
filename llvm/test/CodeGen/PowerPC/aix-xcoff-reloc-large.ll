; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -filetype=obj -code-model=large -o %t.o < %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefixes=RELOC %s
; RUN: llvm-objdump -D -r --symbol-description %t.o | FileCheck --check-prefix=DIS %s

@a = global i32 2, align 4
@b = global i32 10, align 4
@c = global i32 11, align 4

define i32 @foo() {
entry:
  %0 = load i32, i32* @a, align 4
  %1 = load i32, i32* @b, align 4
  %add = add nsw i32 %0, %1
  %2 = load i32, i32* @c, align 4
  %add1 = add nsw i32 %add, %2
  ret i32 %add1
}

; RELOC:        Section (index: {{[0-9]+}}) .text {
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x2
; RELOC-NEXT:     Symbol: a ([[#INDX:]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x6
; RELOC-NEXT:     Symbol: a ([[#INDX]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0xE
; RELOC-NEXT:     Symbol: b ([[#INDX+2]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x12
; RELOC-NEXT:     Symbol: b ([[#INDX+2]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x1A
; RELOC-NEXT:     Symbol: c ([[#INDX+4]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCU (0x30)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x1E
; RELOC-NEXT:     Symbol: c ([[#INDX+4]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOCL (0x31)
; RELOC-NEXT:   }

; DIS:        Disassembly of section .text:
; DIS-EMPTY:
; DIS-NEXT:   00000000 (idx: {{[0-9]+}}) .foo:
; DIS-NEXT:          0: 3c 62 00 00   addis 3, 2, 0
; DIS-NEXT:                           00000002:  R_TOCU       (idx: [[#INDX:]]) a[TE]
; DIS-NEXT:          4: 80 63 00 00   lwz 3, 0(3)
; DIS-NEXT:                           00000006:  R_TOCL       (idx: [[#INDX]]) a[TE]
; DIS-NEXT:          8: 80 63 00 00   lwz 3, 0(3)
; DIS-NEXT:          c: 3c 82 00 00   addis 4, 2, 0
; DIS-NEXT:                           0000000e:  R_TOCU       (idx: [[#INDX+2]]) b[TE]
; DIS-NEXT:         10: 80 84 00 04   lwz 4, 4(4)
; DIS-NEXT:                           00000012:  R_TOCL       (idx: [[#INDX+2]]) b[TE]
; DIS-NEXT:         14: 80 84 00 00   lwz 4, 0(4)
; DIS-NEXT:         18: 3c a2 00 00   addis 5, 2, 0
; DIS-NEXT:                           0000001a:  R_TOCU       (idx: [[#INDX+4]]) c[TE]
; DIS-NEXT:         1c: 80 a5 00 08   lwz 5, 8(5)
; DIS-NEXT:                           0000001e:  R_TOCL       (idx: [[#INDX+4]]) c[TE]
; DIS-NEXT:         20: 7c 63 22 14   add 3, 3, 4
; DIS-NEXT:         24: 80 a5 00 00   lwz 5, 0(5)
; DIS-NEXT:         28: 7c 63 2a 14   add 3, 3, 5
; DIS-NEXT:         2c: 4e 80 00 20   blr
