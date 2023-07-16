; RUN: llc -mtriple powerpc-ibm-aix-xcoff -O0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK32,NOOPT
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -O0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK64,NOOPT

; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefixes=CHECK,CHECK32,OPT
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefixes=CHECK,CHECK64,OPT

; RUN: llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff -verify-machineinstrs < %s -o %t32.o
; RUN: llvm-readobj %t32.o --syms --relocs | FileCheck %s --check-prefix=OBJ32
; RUN: llvm-objdump %t32.o -dr | FileCheck %s --check-prefix=DIS32

; RUN: llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff -verify-machineinstrs < %s -o %t64.o
; RUN: llvm-readobj %t64.o --syms --relocs | FileCheck %s --check-prefix=OBJ64
; RUN: llvm-objdump %t64.o -dr | FileCheck %s --check-prefix=DIS64

@i1 = external constant i32 #0
@i2 = constant ptr @i1 #0

define i32 @read() {
  %1  = load i32, ptr @i1, align 4
  ret i32 %1
}

define ptr @retptr() {
  ret ptr @i2
}

attributes #0 = { "toc-data" }

; CHECK-LABEL: .read:
; NOOPT:         la 3, i1[TD](2)
; NOOPT:         lwz 3, 0(3)
; OPT:           lwz 3, i1[TD](2)

; CHECK-LABEL: .retptr:
; CHECK:         la 3, i2[TD](2)
; CHECK-NEXT:    blr

; CHECK-DAG:   .toc
; CHECK:         .extern i1[TD]
; CHECK32:       .csect i2[TD],2
; CHECK64:       .csect i2[TD],3

; OBJ32:      Relocations [
; OBJ32-NEXT:   Section (index: 1) .text {
; OBJ32-NEXT:     0x2 R_TOC i1(1) 0xF
; OBJ32-NEXT:     0x22 R_TOC i2(15) 0xF
; OBJ32-NEXT:   }
; OBJ32-NEXT:   Section (index: 2) .data {
; OBJ32-NEXT:     0x40 R_POS .read(5) 0x1F
; OBJ32-NEXT:     0x44 R_POS TOC(13) 0x1F
; OBJ32-NEXT:     0x4C R_POS .retptr(7) 0x1F
; OBJ32-NEXT:     0x50 R_POS TOC(13) 0x1F
; OBJ32-NEXT:     0x58 R_POS i1(1) 0x1F
; OBJ32-NEXT:   }
; OBJ32-NEXT: ]

; OBJ32:      Symbol {
; OBJ32:        Index: 1
; OBJ32-NEXT:   Name: i1
; OBJ32-NEXT:   Value (RelocatableAddress): 0x0
; OBJ32-NEXT:   Section: N_UNDEF
; OBJ32-NEXT:   Type: 0x0
; OBJ32-NEXT:   StorageClass: C_EXT (0x2)
; OBJ32-NEXT:   NumberOfAuxEntries: 1
; OBJ32-NEXT:   CSECT Auxiliary Entry {
; OBJ32-NEXT:     Index: 2
; OBJ32-NEXT:     SectionLen: 0
; OBJ32-NEXT:     ParameterHashIndex: 0x0
; OBJ32-NEXT:     TypeChkSectNum: 0x0
; OBJ32-NEXT:     SymbolAlignmentLog2: 0
; OBJ32-NEXT:     SymbolType: XTY_ER (0x0)
; OBJ32-NEXT:     StorageMappingClass: XMC_TD (0x10)
; OBJ32-NEXT:     StabInfoIndex: 0x0
; OBJ32-NEXT:     StabSectNum: 0x0
; OBJ32-NEXT:   }
; OBJ32-NEXT: }
; OBJ32:      Symbol {
; OBJ32:        Index: 13
; OBJ32-NEXT:   Name: TOC
; OBJ32-NEXT:   Value (RelocatableAddress): 0x58
; OBJ32-NEXT:   Section: .data
; OBJ32-NEXT:   Type: 0x0
; OBJ32-NEXT:   StorageClass: C_HIDEXT (0x6B)
; OBJ32-NEXT:   NumberOfAuxEntries: 1
; OBJ32-NEXT:   CSECT Auxiliary Entry {
; OBJ32-NEXT:     Index: 14
; OBJ32-NEXT:     SectionLen: 0
; OBJ32-NEXT:     ParameterHashIndex: 0x0
; OBJ32-NEXT:     TypeChkSectNum: 0x0
; OBJ32-NEXT:     SymbolAlignmentLog2: 2
; OBJ32-NEXT:     SymbolType: XTY_SD (0x1)
; OBJ32-NEXT:     StorageMappingClass: XMC_TC0 (0xF)
; OBJ32-NEXT:     StabInfoIndex: 0x0
; OBJ32-NEXT:     StabSectNum: 0x0
; OBJ32-NEXT:   }
; OBJ32-NEXT: }
; OBJ32:      Symbol {
; OBJ32:        Index: 15
; OBJ32-NEXT:   Name: i2
; OBJ32-NEXT:   Value (RelocatableAddress): 0x58
; OBJ32-NEXT:   Section: .data
; OBJ32-NEXT:   Type: 0x0
; OBJ32-NEXT:   StorageClass: C_EXT (0x2)
; OBJ32-NEXT:   NumberOfAuxEntries: 1
; OBJ32-NEXT:   CSECT Auxiliary Entry {
; OBJ32-NEXT:     Index: 16
; OBJ32-NEXT:     SectionLen: 4
; OBJ32-NEXT:     ParameterHashIndex: 0x0
; OBJ32-NEXT:     TypeChkSectNum: 0x0
; OBJ32-NEXT:     SymbolAlignmentLog2: 2
; OBJ32-NEXT:     SymbolType: XTY_SD (0x1)
; OBJ32-NEXT:     StorageMappingClass: XMC_TD (0x10)
; OBJ32-NEXT:     StabInfoIndex: 0x0
; OBJ32-NEXT:     StabSectNum: 0x0
; OBJ32-NEXT:   }
; OBJ32-NEXT: }

; OBJ64:      Relocations [
; OBJ64-NEXT:   Section (index: 1) .text {
; OBJ64-NEXT:     0x2 R_TOC i1(1) 0xF
; OBJ64-NEXT:     0x22 R_TOC i2(15) 0xF
; OBJ64-NEXT:   }
; OBJ64-NEXT:   Section (index: 2) .data {
; OBJ64-NEXT:     0x40 R_POS .read(5) 0x3F
; OBJ64-NEXT:     0x48 R_POS TOC(13) 0x3F
; OBJ64-NEXT:     0x58 R_POS .retptr(7) 0x3F
; OBJ64-NEXT:     0x60 R_POS TOC(13) 0x3F
; OBJ64-NEXT:     0x70 R_POS i1(1) 0x3F
; OBJ64-NEXT:   }
; OBJ64-NEXT: ]

; OBJ64:      Symbol {
; OBJ64:        Index: 1
; OBJ64-NEXT:   Name: i1
; OBJ64-NEXT:   Value (RelocatableAddress): 0x0
; OBJ64-NEXT:   Section: N_UNDEF
; OBJ64-NEXT:   Type: 0x0
; OBJ64-NEXT:   StorageClass: C_EXT (0x2)
; OBJ64-NEXT:   NumberOfAuxEntries: 1
; OBJ64-NEXT:   CSECT Auxiliary Entry {
; OBJ64-NEXT:     Index: 2
; OBJ64-NEXT:     SectionLen: 0
; OBJ64-NEXT:     ParameterHashIndex: 0x0
; OBJ64-NEXT:     TypeChkSectNum: 0x0
; OBJ64-NEXT:     SymbolAlignmentLog2: 0
; OBJ64-NEXT:     SymbolType: XTY_ER (0x0)
; OBJ64-NEXT:     StorageMappingClass: XMC_TD (0x10)
; OBJ64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:   }
; OBJ64-NEXT: }
; OBJ64:      Symbol {
; OBJ64:        Index: 13
; OBJ64-NEXT:   Name: TOC
; OBJ64-NEXT:   Value (RelocatableAddress): 0x70
; OBJ64-NEXT:   Section: .data
; OBJ64-NEXT:   Type: 0x0
; OBJ64-NEXT:   StorageClass: C_HIDEXT (0x6B)
; OBJ64-NEXT:   NumberOfAuxEntries: 1
; OBJ64-NEXT:   CSECT Auxiliary Entry {
; OBJ64-NEXT:     Index: 14
; OBJ64-NEXT:     SectionLen: 0
; OBJ64-NEXT:     ParameterHashIndex: 0x0
; OBJ64-NEXT:     TypeChkSectNum: 0x0
; OBJ64-NEXT:     SymbolAlignmentLog2: 2
; OBJ64-NEXT:     SymbolType: XTY_SD (0x1)
; OBJ64-NEXT:     StorageMappingClass: XMC_TC0 (0xF)
; OBJ64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:   }
; OBJ64-NEXT: }
; OBJ64:      Symbol {
; OBJ64:        Index: 15
; OBJ64-NEXT:   Name: i2
; OBJ64-NEXT:   Value (RelocatableAddress): 0x70
; OBJ64-NEXT:   Section: .data
; OBJ64-NEXT:   Type: 0x0
; OBJ64-NEXT:   StorageClass: C_EXT (0x2)
; OBJ64-NEXT:   NumberOfAuxEntries: 1
; OBJ64-NEXT:   CSECT Auxiliary Entry {
; OBJ64-NEXT:     Index: 16
; OBJ64-NEXT:     SectionLen: 8
; OBJ64-NEXT:     ParameterHashIndex: 0x0
; OBJ64-NEXT:     TypeChkSectNum: 0x0
; OBJ64-NEXT:     SymbolAlignmentLog2: 3
; OBJ64-NEXT:     SymbolType: XTY_SD (0x1)
; OBJ64-NEXT:     StorageMappingClass: XMC_TD (0x10)
; OBJ64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:   }
; OBJ64-NEXT: }

; DIS32:      00000000 <.read>:
; DIS32-NEXT:        0: 80 62 00 00   lwz 3, 0(2)
; DIS32-NEXT:                         00000002:  R_TOC	i1
; DIS32:      00000020 <.retptr>:
; DIS32-NEXT:       20: 38 62 00 00  	addi 3, 2, 0
; DIS32-NEXT:                         00000022:  R_TOC	i2

; DIS64:      0000000000000000 <.read>:
; DIS64-NEXT:        0: 80 62 00 00  	lwz 3, 0(2)
; DIS64-NEXT:                         0000000000000002:  R_TOC	i1
; DIS64:      0000000000000020 <.retptr>:
; DIS64-NEXT:       20: 38 62 00 00  	addi 3, 2, 0
; DIS64-NEXT:                         0000000000000022:  R_TOC	i2
