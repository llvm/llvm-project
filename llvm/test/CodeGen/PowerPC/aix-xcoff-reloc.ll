; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -mattr=-altivec \
; RUN:     -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --section-headers --file-header %t.o | FileCheck --check-prefixes=OBJ,OBJ32 %s
; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck --check-prefixes=RELOC,RELOC32 %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYM,SYM32 %s
; RUN: llvm-objdump -D %t.o | FileCheck --check-prefix=DIS %s
; RUN: llvm-objdump -r %t.o | FileCheck --check-prefix=DIS_REL %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff -mattr=-altivec \
; RUN:     -xcoff-traceback-table=false -data-sections=false -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --section-headers --file-header %t64.o | FileCheck --check-prefixes=OBJ,OBJ64 %s
; RUN: llvm-readobj --relocs --expand-relocs %t64.o | FileCheck --check-prefixes=RELOC,RELOC64 %s
; RUN: llvm-readobj --syms %t64.o | FileCheck --check-prefixes=SYM,SYM64 %s
; RUN: llvm-objdump -D %t64.o | FileCheck --check-prefix=DIS64 %s
; RUN: llvm-objdump -r %t64.o | FileCheck --check-prefix=DIS_REL64 %s

@globalA = global i32 1, align 4
@globalB = global i32 2, align 4
@arr = global <{ i32, [9 x i32] }> <{ i32 3, [9 x i32] zeroinitializer }>, align 4
@p = global ptr getelementptr (i8, ptr @arr, i64 16), align 4

define i32 @foo() {
entry:
  %call = call i32 @bar(i32 1)
  %0 = load i32, ptr @globalA, align 4
  %add = add nsw i32 %call, %0
  %1 = load i32, ptr @globalB, align 4
  %add1 = add nsw i32 %add, %1
  ret i32 %add1
}

declare i32 @bar(i32)

; OBJ:        FileHeader {
; OBJ32-NEXT:   Magic: 0x1DF
; OBJ64-NEXT:   Magic: 0x1F7
; OBJ-NEXT:     NumberOfSections: 2
; OBJ-NEXT:     TimeStamp: None (0x0)
; OBJ32-NEXT:   SymbolTableOffset: 0x13C
; OBJ64-NEXT:   SymbolTableOffset: 0x1B8
; OBJ-NEXT:     SymbolTableEntries: 27
; OBJ-NEXT:     OptionalHeaderSize: 0x0
; OBJ-NEXT:     Flags: 0x0
; OBJ-NEXT:   }
; OBJ-NEXT: Sections [
; OBJ-NEXT:   Section {
; OBJ-NEXT:     Index: 1
; OBJ-NEXT:     Name: .text
; OBJ-NEXT:     PhysicalAddress: 0x0
; OBJ-NEXT:     VirtualAddress: 0x0
; OBJ-NEXT:     Size: 0x40
; OBJ32-NEXT:   RawDataOffset: 0x64
; OBJ32-NEXT:   RelocationPointer: 0xEC
; OBJ64-NEXT:   RawDataOffset: 0xA8
; OBJ64-NEXT:   RelocationPointer: 0x148
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
; OBJ32-NEXT:   Size: 0x48
; OBJ32-NEXT:   RawDataOffset: 0xA4
; OBJ32-NEXT:   RelocationPointer: 0x10A
; OBJ64-NEXT:   Size: 0x60
; OBJ64-NEXT:   RawDataOffset: 0xE8
; OBJ64-NEXT:   RelocationPointer: 0x172
; OBJ-NEXT:     LineNumberPointer: 0x0
; OBJ-NEXT:     NumberOfRelocations: 5
; OBJ-NEXT:     NumberOfLineNumbers: 0
; OBJ-NEXT:     Type: STYP_DATA (0x40)
; OBJ-NEXT:   }
; OBJ-NEXT: ]

; RELOC:      Relocations [
; RELOC-NEXT:   Section (index: 1) .text {
; RELOC-NEXT:     Relocation {
; RELOC-NEXT:       Virtual Address: 0x10
; RELOC-NEXT:       Symbol: .bar (1)
; RELOC-NEXT:       IsSigned: Yes
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 26
; RELOC-NEXT:       Type: R_RBR (0x1A)
; RELOC-NEXT:     }
; RELOC-NEXT:     Relocation {
; RELOC-NEXT:       Virtual Address: 0x1A
; RELOC-NEXT:       Symbol: globalA (23)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC-NEXT:     Relocation {
; RELOC-NEXT:       Virtual Address: 0x1E
; RELOC-NEXT:       Symbol: globalB (25)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC-NEXT:       Length: 16
; RELOC-NEXT:       Type: R_TOC (0x3)
; RELOC-NEXT:     }
; RELOC-NEXT:   }
; RELOC-NEXT:   Section (index: 2) .data {
; RELOC-NEXT:     Relocation {
; RELOC-NEXT:       Virtual Address: 0x70
; RELOC-NEXT:       Symbol: arr (15)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC32-NEXT:     Length: 32
; RELOC64-NEXT:     Length: 64
; RELOC-NEXT:       Type: R_POS (0x0)
; RELOC-NEXT:     }
; RELOC-NEXT:     Relocation {
; RELOC32-NEXT:     Virtual Address: 0x74
; RELOC64-NEXT:     Virtual Address: 0x78
; RELOC-NEXT:       Symbol: .foo (7)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC32-NEXT:     Length: 32
; RELOC64-NEXT:     Length: 64
; RELOC-NEXT:       Type: R_POS (0x0)
; RELOC-NEXT:     }
; RELOC-NEXT:     Relocation {
; RELOC32-NEXT:     Virtual Address: 0x78
; RELOC64-NEXT:     Virtual Address: 0x80
; RELOC-NEXT:       Symbol: TOC (21)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC32-NEXT:     Length: 32
; RELOC64-NEXT:     Length: 64
; RELOC-NEXT:       Type: R_POS (0x0)
; RELOC-NEXT:     }
; RELOC-NEXT:     Relocation {
; RELOC32-NEXT:     Virtual Address: 0x80
; RELOC64-NEXT:     Virtual Address: 0x90
; RELOC-NEXT:       Symbol: globalA (11)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC32-NEXT:     Length: 32
; RELOC64-NEXT:     Length: 64
; RELOC-NEXT:       Type: R_POS (0x0)
; RELOC-NEXT:     }
; RELOC-NEXT:     Relocation {
; RELOC32-NEXT:     Virtual Address: 0x84
; RELOC64-NEXT:     Virtual Address: 0x98
; RELOC-NEXT:       Symbol: globalB (13)
; RELOC-NEXT:       IsSigned: No
; RELOC-NEXT:       FixupBitValue: 0
; RELOC32-NEXT:     Length: 32
; RELOC64-NEXT:     Length: 64
; RELOC-NEXT:       Type: R_POS (0x0)
; RELOC-NEXT:     }
; RELOC-NEXT:   }
; RELOC-NEXT: ]

; SYM:      Symbols [
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: 0
; SYM-NEXT:     Name: <stdin>
; SYM-NEXT:     Value (SymbolTableIndex): 0x0
; SYM-NEXT:     Section: N_DEBUG
; SYM-NEXT:     Source Language ID: TB_CPLUSPLUS (0x9)
; SYM32-NEXT:   CPU Version ID: TCPU_COM (0x3)
; SYM64-NEXT:   CPU Version ID: TCPU_PPC64 (0x2)
; SYM-NEXT:     StorageClass: C_FILE (0x67)
; SYM-NEXT:     NumberOfAuxEntries: 0
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX:]]
; SYM-NEXT:     Name: .bar
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+1]]
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_ER (0x0)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+2]]
; SYM-NEXT:     Name: bar
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: N_UNDEF
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+3]]
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_ER (0x0)
; SYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+4]]
; SYM-NEXT:     Name:
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+5]]
; SYM-NEXT:       SectionLen: 64
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 5
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+6]]
; SYM-NEXT:     Name: .foo
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .text
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+7]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#INDX+4]]
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_PR (0x0)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+8]]
; SYM-NEXT:     Name: .data
; SYM-NEXT:     Value (RelocatableAddress): 0x40
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+9]]
; SYM32-NEXT:     SectionLen: 52
; SYM64-NEXT:     SectionLen: 56
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:     SymbolAlignmentLog2: 2
; SYM64-NEXT:     SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+10]]
; SYM-NEXT:     Name: globalA
; SYM-NEXT:     Value (RelocatableAddress): 0x40
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+11]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#INDX+8]]
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+12]]
; SYM-NEXT:     Name: globalB
; SYM-NEXT:     Value (RelocatableAddress): 0x44
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+13]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#INDX+8]]
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+14]]
; SYM-NEXT:     Name: arr
; SYM-NEXT:     Value (RelocatableAddress): 0x48
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+15]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#INDX+8]]
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+16]]
; SYM-NEXT:     Name: p
; SYM-NEXT:     Value (RelocatableAddress): 0x70
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+17]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#INDX+8]]
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 0
; SYM-NEXT:       SymbolType: XTY_LD (0x2)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+18]]
; SYM-NEXT:     Name: foo
; SYM32-NEXT:   Value (RelocatableAddress): 0x74
; SYM64-NEXT:   Value (RelocatableAddress): 0x78
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+19]]
; SYM32-NEXT:     SectionLen: 12
; SYM64-NEXT:     SectionLen: 24
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:     SymbolAlignmentLog2: 2
; SYM64-NEXT:     SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_DS (0xA)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+20]]
; SYM-NEXT:     Name: TOC
; SYM32-NEXT:   Value (RelocatableAddress): 0x80
; SYM64-NEXT:   Value (RelocatableAddress): 0x90
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+21]]
; SYM-NEXT:       SectionLen: 0
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC0 (0xF)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+22]]
; SYM-NEXT:     Name: globalA
; SYM32-NEXT:   Value (RelocatableAddress): 0x80
; SYM64-NEXT:   Value (RelocatableAddress): 0x90
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+23]]
; SYM32-NEXT:     SectionLen: 4
; SYM64-NEXT:     SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:     SymbolAlignmentLog2: 2
; SYM64-NEXT:     SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT:   Symbol {
; SYM-NEXT:     Index: [[#INDX+24]]
; SYM-NEXT:     Name: globalB
; SYM32-NEXT:   Value (RelocatableAddress): 0x84
; SYM64-NEXT:   Value (RelocatableAddress): 0x98
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+25]]
; SYM32-NEXT:     SectionLen: 4
; SYM64-NEXT:     SectionLen: 8
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM32-NEXT:     SymbolAlignmentLog2: 2
; SYM64-NEXT:     SymbolAlignmentLog2: 3
; SYM-NEXT:       SymbolType: XTY_SD (0x1)
; SYM-NEXT:       StorageMappingClass: XMC_TC (0x3)
; SYM32-NEXT:     StabInfoIndex: 0x0
; SYM32-NEXT:     StabSectNum: 0x0
; SYM64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; SYM-NEXT:     }
; SYM-NEXT:   }
; SYM-NEXT: ]


; DIS:      {{.*}}aix-xcoff-reloc.ll.tmp.o:   file format aixcoff-rs6000
; DIS:      Disassembly of section .text:
; DIS:      00000000 <.foo>:
; DIS-NEXT:        0: 7c 08 02 a6                   mflr 0
; DIS-NEXT:        4: 94 21 ff c0                   stwu 1, -64(1)
; DIS-NEXT:        8: 38 60 00 01                   li 3, 1
; DIS-NEXT:        c: 90 01 00 48                   stw 0, 72(1)
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

; DIS64:      Disassembly of section .text:
; DIS64:      0000000000000000 <.foo>:
; DIS64-NEXT:        0: 7c 08 02 a6  	mflr 0
; DIS64-NEXT:        4: f8 21 ff 91  	stdu 1, -112(1)
; DIS64-NEXT:        8: 38 60 00 01  	li 3, 1
; DIS64-NEXT:        c: f8 01 00 80  	std 0, 128(1)
; DIS64-NEXT:       10: 4b ff ff f1  	bl 0x0 <.foo>
; DIS64-NEXT:       14: 60 00 00 00  	nop
; DIS64-NEXT:       18: e8 82 00 00  	ld 4, 0(2)
; DIS64-NEXT:       1c: e8 a2 00 08  	ld 5, 8(2)
; DIS64-NEXT:       20: 80 84 00 00  	lwz 4, 0(4)
; DIS64-NEXT:       24: 80 a5 00 00  	lwz 5, 0(5)
; DIS64-NEXT:       28: 7c 63 22 14  	add 3, 3, 4
; DIS64-NEXT:       2c: 7c 63 2a 14  	add 3, 3, 5
; DIS64-NEXT:       30: 38 21 00 70  	addi 1, 1, 112
; DIS64-NEXT:       34: e8 01 00 10  	ld 0, 16(1)
; DIS64-NEXT:       38: 7c 08 03 a6  	mtlr 0
; DIS64-NEXT:       3c: 4e 80 00 20  	blr

; DIS64:      Disassembly of section .data:
; DIS64:      0000000000000040 <globalA>:
; DIS64-NEXT:       40: 00 00 00 01  	<unknown>
; DIS64:      0000000000000044 <globalB>:
; DIS64-NEXT:       44: 00 00 00 02  	<unknown>
; DIS64:      0000000000000048 <arr>:
; DIS64-NEXT:       48: 00 00 00 03  	<unknown>
; DIS64-NEXT: 		...
; DIS64:      0000000000000070 <p>:
; DIS64-NEXT:       70: 00 00 00 00  	<unknown>
; DIS64-NEXT:       74: 00 00 00 58  	<unknown>
; DIS64:      0000000000000078 <foo>:
; DIS64-NEXT: 		...
; DIS64-NEXT:       84: 00 00 00 90  	<unknown>
; DIS64-NEXT: 		...
; DIS64:      0000000000000090 <globalA>:
; DIS64-NEXT:       90: 00 00 00 00  	<unknown>
; DIS64-NEXT:       94: 00 00 00 40  	<unknown>
; DIS64:      0000000000000098 <globalB>:
; DIS64-NEXT:       98: 00 00 00 00  	<unknown>
; DIS64-NEXT:       9c: 00 00 00 44  	<unknown>

; DIS_REL64:      RELOCATION RECORDS FOR [.text]:
; DIS_REL64-NEXT: OFFSET           TYPE                     VALUE
; DIS_REL64-NEXT: 0000000000000010 R_RBR                    .bar
; DIS_REL64-NEXT: 000000000000001a R_TOC                    globalA
; DIS_REL64-NEXT: 000000000000001e R_TOC                    globalB

; DIS_REL64:      RELOCATION RECORDS FOR [.data]:
; DIS_REL64-NEXT: OFFSET           TYPE                     VALUE
; DIS_REL64-NEXT: 0000000000000030 R_POS                    arr
; DIS_REL64-NEXT: 0000000000000038 R_POS                    .foo
; DIS_REL64-NEXT: 0000000000000040 R_POS                    TOC
; DIS_REL64-NEXT: 0000000000000050 R_POS                    globalA
; DIS_REL64-NEXT: 0000000000000058 R_POS                    globalB
