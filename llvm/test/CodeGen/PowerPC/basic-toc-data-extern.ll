; RUN: llc -mtriple powerpc-ibm-aix-xcoff -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -verify-machineinstrs < %s | FileCheck %s

; RUN: llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff -verify-machineinstrs < %s -o %t32.o
; RUN: llvm-readobj %t32.o --syms --relocs | FileCheck %s --check-prefix=OBJ32
; RUN: llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff -verify-machineinstrs < %s -o %t64.o
; RUN: llvm-readobj %t64.o --syms --relocs | FileCheck %s --check-prefix=OBJ64

@i = external global i32, align 4  #0

; Function Attrs: noinline nounwind optnone
define ptr @get() {
  entry:
    ret ptr @i
}

attributes #0 = { "toc-data" }

; CHECK:        la 3, i[TD](2)
; CHECK:        .toc
; CHECK-NEXT:   .extern i[TD]

; OBJ32:      Relocations [
; OBJ32-NEXT:   Section (index: 1) .text {
; OBJ32-NEXT:     0x2 R_TOC i(1) 0xF
; OBJ32-NEXT:   }
; OBJ32-NEXT:   Section (index: 2) .data {
; OBJ32-NEXT:     0x20 R_POS .get(5) 0x1F
; OBJ32-NEXT:     0x24 R_POS TOC(9) 0x1F
; OBJ32-NEXT:   }
; OBJ32-NEXT: ]

; OBJ32:      Symbol {
; OBJ32:        Index: 1
; OBJ32-NEXT:   Name: i
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
; OBJ32:        Index: 9
; OBJ32-NEXT:   Name: TOC
; OBJ32-NEXT:   Value (RelocatableAddress): 0x2C
; OBJ32-NEXT:   Section: .data
; OBJ32-NEXT:   Type: 0x0
; OBJ32-NEXT:   StorageClass: C_HIDEXT (0x6B)
; OBJ32-NEXT:   NumberOfAuxEntries: 1
; OBJ32-NEXT:   CSECT Auxiliary Entry {
; OBJ32-NEXT:     Index: 10
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

; OBJ64:      Relocations [
; OBJ64-NEXT:   Section (index: 1) .text {
; OBJ64-NEXT:     0x2 R_TOC i(1) 0xF
; OBJ64-NEXT:   }
; OBJ64-NEXT:   Section (index: 2) .data {
; OBJ64-NEXT:     0x20 R_POS .get(5) 0x3F
; OBJ64-NEXT:     0x28 R_POS TOC(9) 0x3F
; OBJ64-NEXT:   }
; OBJ64-NEXT: ]

; OBJ64:      Symbol {
; OBJ64:        Index: 1
; OBJ64-NEXT:   Name: i
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
; OBJ64:        Index: 9
; OBJ64-NEXT:   Name: TOC
; OBJ64-NEXT:   Value (RelocatableAddress): 0x38
; OBJ64-NEXT:   Section: .data
; OBJ64-NEXT:   Type: 0x0
; OBJ64-NEXT:   StorageClass: C_HIDEXT (0x6B)
; OBJ64-NEXT:   NumberOfAuxEntries: 1
; OBJ64-NEXT:   CSECT Auxiliary Entry {
; OBJ64-NEXT:     Index: 10
; OBJ64-NEXT:     SectionLen: 0
; OBJ64-NEXT:     ParameterHashIndex: 0x0
; OBJ64-NEXT:     TypeChkSectNum: 0x0
; OBJ64-NEXT:     SymbolAlignmentLog2: 2
; OBJ64-NEXT:     SymbolType: XTY_SD (0x1)
; OBJ64-NEXT:     StorageMappingClass: XMC_TC0 (0xF)
; OBJ64-NEXT:     Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:   }
; OBJ64-NEXT: }
