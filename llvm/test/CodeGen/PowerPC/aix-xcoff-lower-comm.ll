; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -data-sections=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,ASM32 %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff -data-sections=false < %s | \
; RUN:   FileCheck --check-prefixes=CHECK,ASM64 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -data-sections=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj -r --expand-relocs --syms %t.o | FileCheck --check-prefixes=RELOC,SYM %s

; RUN: not --crash llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -data-sections=false -filetype=obj < %s 2>&1 | \
; RUN:   FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@common = common global i32 0, align 4
@pointer = global i32* @common, align 4


; CHECK:             .comm   common[RW],4,2
; ASM32-NEXT:        .csect .data[RW],2
; ASM64-NEXT:        .csect .data[RW],3
; CHECK-NEXT:        .globl  pointer
; ASM32-NEXT:        .align  2
; ASM64-NEXT:        .align  3
; CHECK-NEXT:pointer:
; ASM32-NEXT:        .vbyte	4, common[RW]
; ASM64-NEXT:        .vbyte	8, common[RW]


; RELOC:      Relocations [
; RELOC-NEXT:   Section (index: {{[0-9]+}}) .data {
; RELOC-NEXT:   Relocation {
; RELOC-NEXT:     Virtual Address: 0x0
; RELOC-NEXT:     Symbol: common ([[#COM_INDX:]])
; RELOC-NEXT:     IsSigned: No
; RELOC-NEXT:     FixupBitValue: 0
; RELOC-NEXT:     Length: 32
; RELOC-NEXT:     Type: R_POS (0x0)
; RELOC-NEXT:   }
; RELOC-NEXT: }
; RELOC-NEXT: ]

; SYM:        Symbol {{[{][[:space:]] *}}Index: [[#INDX:]]{{[[:space:]] *}}Name: .data
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_HIDEXT (0x6B)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+1]]
; SYM-NEXT:       SectionLen: 4
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
; SYM-NEXT:     Index: [[#INDX+2]]
; SYM-NEXT:     Name: pointer
; SYM-NEXT:     Value (RelocatableAddress): 0x0
; SYM-NEXT:     Section: .data
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#INDX+3]]
; SYM-NEXT:       ContainingCsectSymbolIndex: [[#INDX]]
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
; SYM-NEXT:     Index: [[#COM_INDX]]
; SYM-NEXT:     Name: common
; SYM-NEXT:     Value (RelocatableAddress): 0x4
; SYM-NEXT:     Section: .bss
; SYM-NEXT:     Type: 0x0
; SYM-NEXT:     StorageClass: C_EXT (0x2)
; SYM-NEXT:     NumberOfAuxEntries: 1
; SYM-NEXT:     CSECT Auxiliary Entry {
; SYM-NEXT:       Index: [[#COM_INDX+1]]
; SYM-NEXT:       SectionLen: 4
; SYM-NEXT:       ParameterHashIndex: 0x0
; SYM-NEXT:       TypeChkSectNum: 0x0
; SYM-NEXT:       SymbolAlignmentLog2: 2
; SYM-NEXT:       SymbolType: XTY_CM (0x3)
; SYM-NEXT:       StorageMappingClass: XMC_RW (0x5)
; SYM-NEXT:       StabInfoIndex: 0x0
; SYM-NEXT:       StabSectNum: 0x0
; SYM-NEXT:     }
; SYM-NEXT:   }
