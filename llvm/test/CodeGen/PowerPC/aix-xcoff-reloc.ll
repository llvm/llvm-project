; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -mattr=-altivec \
; RUN:     -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | \
; RUN: FileCheck --check-prefix=OBJ %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefix=RELOC %s
; RUN: llvm-readobj -t %t.o | FileCheck --check-prefix=SYM %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=DIS %s
; RUN: llvm-objdump -r %t.o | FileCheck --check-prefix=DIS_REL %s

; RUN: not --crash llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff -mattr=-altivec -data-sections=false -filetype=obj < %s 2>&1 | \
; RUN: FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@globalA = global i32 1, align 4
@globalB = global i32 2, align 4
@arr = global <{ i32, [9 x i32] }> <{ i32 3, [9 x i32] zeroinitializer }>, align 4
@p = global i32* bitcast (i8* getelementptr (i8, i8* bitcast (<{ i32, [9 x i32] }>* @arr to i8*), i64 16) to i32*), align 4

define i32 @foo() {
entry:
  %call = call i32 @bar(i32 1)
  %0 = load i32, i32* @globalA, align 4
  %add = add nsw i32 %call, %0
  %1 = load i32, i32* @globalB, align 4
  %add1 = add nsw i32 %add, %1
  ret i32 %add1
}

declare i32 @bar(i32)

; OBJ:      File: {{.*}}aix-xcoff-reloc.ll.tmp.o
; OBJ-NEXT: Format: aixcoff-rs6000
; OBJ-NEXT: Arch: powerpc
; OBJ-NEXT: AddressSize: 32bit
; OBJ-NEXT: FileHeader {
; OBJ-NEXT:   Magic: 0x1DF
; OBJ-NEXT:   NumberOfSections: 2
; OBJ-NEXT:   TimeStamp: None (0x0)
; OBJ-NEXT:   SymbolTableOffset: 0x13C
; OBJ-NEXT:   SymbolTableEntries: 27
; OBJ-NEXT:   OptionalHeaderSize: 0x0
; OBJ-NEXT:   Flags: 0x0
; OBJ-NEXT: }
; OBJ-NEXT: Sections [
; OBJ-NEXT:   Section {
; OBJ-NEXT:     Index: 1
; OBJ-NEXT:     Name: .text
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x40
; OBJ-NEXT:     RawDataOffset: 0x64
; OBJ-NEXT:     RelocationPointer: 0xEC
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 3
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_TEXT (0x20)
; OBJ-NEXT:   }
; OBJ-NEXT:   Section {
; OBJ-NEXT:     Index: 2
; OBJ-NEXT:     Name: .data
; OBJ-NEXT:     PhysicalAddress: 0x40
; OBJ-NEXT:     VirtualAddress: 0x40
; OBJ-NEXT:     Size: 0x48
; OBJ-NEXT:     RawDataOffset: 0xA4
; OBJ-NEXT:     RelocationPointer: 0x10A
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 5
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_DATA (0x40)
; OBJ-NEXT:   }
; OBJ-NEXT: ]


; RELOC:      File: {{.*}}aix-xcoff-reloc.ll.tmp.o
; RELOC-NEXT: Format: aixcoff-rs6000
; RELOC-NEXT: Arch: powerpc
; RELOC-NEXT: AddressSize: 32bit
; RELOC-NEXT: Relocations [
; RELOC-NEXT:   Section (index: 1) .text {
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x10
; RELOC-NEXT:     Symbol: .bar (1)
; RELOC-NEXT:     IsSigned: Yes
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 26
; RELOC-NEXT:     Type: R_RBR (0x1A)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x1A
; RELOC-NEXT:     Symbol: globalA (23)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x1E
; RELOC-NEXT:     Symbol: globalB (25)
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 16
; RELOC-NEXT:     Type: R_TOC (0x3)
; RELOC-NEXT:   }
; RELOC-NEXT: }
; RELOC-NEXT: Section (index: 2) .data {
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x70
; RELOC-NEXT:   Symbol: arr (15)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x74
; RELOC-NEXT:   Symbol: .foo (7)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x78
; RELOC-NEXT:   Symbol: TOC (21)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x80
; RELOC-NEXT:   Symbol: globalA (11)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: Relocation {
; RELOC-NEXT:   Virtual Address: 0x84
; RELOC-NEXT:   Symbol: globalB (13)
; RELOC-NEXT:   IsSigned: No
; RELOC-NEXT:   FixupBitValue: 0
; RELOC-NEXT:   Length: 32
; RELOC-NEXT:   Type: R_POS (0x0)
; RELOC-NEXT: }
; RELOC-NEXT: }
; RELOC-NEXT: ]

; SYM:      Symbols [
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 0
; SYM-NEXT:     Name: .file
; SYM-NEXT:     Value (SymbolTableIndex): 0x0
; SYM-NEXT:     Section: N_DEBUG
; SYM-NEXT:     Source Language ID: TB_C (0x0)
; SYM-NEXT:     CPU Version ID: 0x0
; SYM-NEXT:     StorageClass: C_FILE (0x67)
; SYM-NEXT:     NumberOfAuxEntries: 0
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 1
; SYM-NEXT:     Name: .bar
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 2
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_ER (0x0)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 3
; SYM-NEXT:     Name: bar
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 4
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_ER (0x0)
; SYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 5
; SYM-NEXT:     Name: .text
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 6
; SYM-NEXT:       SectionLen: 64
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 4
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 7
; SYM-NEXT:     Name: .foo
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 8
; SYM-NEXT:       ContainingCsectSymbolIndex: 5
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 9
; SYM-NEXT:     Name: .data
; SYM-NEXT:     Value (RelocatableAddress): 0x40
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 10
; SYM-NEXT:       SectionLen: 52
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 11
; SYM-NEXT:     Name: globalA
; SYM-NEXT:     Value (RelocatableAddress): 0x40
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 12
; SYM-NEXT:       ContainingCsectSymbolIndex: 9
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 13
; SYM-NEXT:     Name: globalB
; SYM-NEXT:     Value (RelocatableAddress): 0x44
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 14
; SYM-NEXT:       ContainingCsectSymbolIndex: 9
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 15
; SYM-NEXT:     Name: arr
; SYM-NEXT:     Value (RelocatableAddress): 0x48
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 16
; SYM-NEXT:       ContainingCsectSymbolIndex: 9
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 17
; SYM-NEXT:     Name: p
; SYM-NEXT:     Value (RelocatableAddress): 0x70
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 18
; SYM-NEXT:       ContainingCsectSymbolIndex: 9
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 19
; SYM-NEXT:     Name: foo
; SYM-NEXT:     Value (RelocatableAddress): 0x74
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 20
; SYM-NEXT:       SectionLen: 12
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 21
; SYM-NEXT:     Name: TOC
; SYM-NEXT:     Value (RelocatableAddress): 0x80
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 22
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC0 (0xF)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 23
; SYM-NEXT:     Name: globalA
; SYM-NEXT:     Value (RelocatableAddress): 0x80
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 24
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 25
; SYM-NEXT:     Name: globalB
; SYM-NEXT:     Value (RelocatableAddress): 0x84
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: 26
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT: ]


; DIS:      {{.*}}aix-xcoff-reloc.ll.tmp.o:   file format aixcoff-rs6000
; DIS:      Disassembly of section .text:
; DIS:      00000000 <.text>:
; DIS-NEXT:        0: 7c 08 02 a6                   mflr 0
; DIS-NEXT:        4: 90 01 00 08                   stw 0, 8(1)
; DIS-NEXT:        8: 94 21 ff c0                   stwu 1, -64(1)
; DIS-NEXT:        c: 38 60 00 01                   li 3, 1
; DIS-NEXT:       10: 4b ff ff f1                   bl 0x0
; DIS-NEXT:       14: 60 00 00 00                   nop
; DIS-NEXT:       18: 80 82 00 00                   lwz 4, 0(2)
; DIS-NEXT:       1c: 80 a2 00 04                   lwz 5, 4(2)
; DIS-NEXT:       20: 80 84 00 00                   lwz 4, 0(4)
; DIS-NEXT:       24: 80 a5 00 00                   lwz 5, 0(5)
; DIS-NEXT:       28: 7c 63 22 14                   add 3, 3, 4
; DIS-NEXT:       2c: 7c 63 2a 14                   add 3, 3, 5
; DIS-NEXT:       30: 38 21 00 40                   addi 1, 1, 64
; DIS-NEXT:       34: 80 01 00 08                   lwz 0, 8(1)
; DIS-NEXT:       38: 7c 08 03 a6                   mtlr 0
; DIS-NEXT:       3c: 4e 80 00 20                   blr

; DIS:      Disassembly of section .data:
; DIS:      00000040 <globalA>:
; DIS-NEXT:       40: 00 00 00 01                   <unknown>
; DIS:      00000044 <globalB>:
; DIS-NEXT:       44: 00 00 00 02                   <unknown>
; DIS:      00000048 <arr>:
; DIS-NEXT:       48: 00 00 00 03                   <unknown>
; DIS-NEXT:                 ...
; DIS:      00000070 <p>:
; DIS-NEXT:       70: 00 00 00 58                   <unknown>
; DIS:      00000074 <foo>:
; DIS-NEXT:       74: 00 00 00 00                   <unknown>
; DIS-NEXT:       78: 00 00 00 80                   <unknown>
; DIS-NEXT:       7c: 00 00 00 00                   <unknown>
; DIS:      00000080 <globalA>:
; DIS-NEXT:       80: 00 00 00 40                   <unknown>
; DIS:      00000084 <globalB>:
; DIS-NEXT:       84: 00 00 00 44                   <unknown>

; DIS_REL:       {{.*}}aix-xcoff-reloc.ll.tmp.o:   file format aixcoff-rs6000
; DIS_REL:       RELOCATION RECORDS FOR [.text]:
; DIS_REL-NEXT:  OFFSET   TYPE                     VALUE
; DIS_REL-NEXT:  00000010 R_RBR                    .bar
; DIS_REL-NEXT:  0000001a R_TOC                    globalA
; DIS_REL-NEXT:  0000001e R_TOC                    globalB
; DIS_REL:       RELOCATION RECORDS FOR [.data]:
; DIS_REL-NEXT:  OFFSET   TYPE                     VALUE
; DIS_REL-NEXT:  00000030 R_POS                    arr
; DIS_REL-NEXT:  00000034 R_POS                    .foo
; DIS_REL-NEXT:  00000038 R_POS                    TOC
; DIS_REL-NEXT:  00000040 R_POS                    globalA
; DIS_REL-NEXT:  00000044 R_POS                    globalB
