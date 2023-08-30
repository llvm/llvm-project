; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s --check-prefix CHECK
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mxcoff-roptr < %s | FileCheck %s --check-prefix CHECK-RO
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mxcoff-roptr < %s | FileCheck %s --check-prefix CHECK-RO

; RUN: llc -filetype=obj -mtriple powerpc-ibm-aix-xcoff -verify-machineinstrs < %s -o %t32.o
; RUN: llvm-readobj %t32.o --syms --relocs | FileCheck %s --check-prefix=OBJ32

; RUN: llc -filetype=obj -mtriple powerpc64-ibm-aix-xcoff -verify-machineinstrs < %s -o %t64.o
; RUN: llvm-readobj %t64.o --syms --relocs | FileCheck %s --check-prefix=OBJ64

@var = external constant i32
@ptr = private constant ptr @var, section "relro-section"

; CHECK: .extern var[UA]
; CHECK: .csect _Renamed..2drelro_section[RW]

; CHECK-RO: .extern var[UA]
; CHECK-RO: .csect _Renamed..2drelro_section[RO]

; OBJ32:      Relocations [
; OBJ32-NEXT:   Section (index: 2) .data {
; OBJ32-NEXT:     0x0 R_POS var(1) 0x1F
; OBJ32-NEXT:   }
; OBJ32-NEXT: ]
; OBJ32-NEXT: Symbols [
; OBJ32:          Index: 1
; OBJ32-NEXT:     Name: var
; OBJ32-NEXT:     Value (RelocatableAddress): 0x0
; OBJ32-NEXT:     Section: N_UNDEF
; OBJ32-NEXT:     Type: 0x0
; OBJ32-NEXT:     StorageClass: C_EXT (0x2)
; OBJ32-NEXT:     NumberOfAuxEntries: 1
; OBJ32-NEXT:     CSECT Auxiliary Entry {
; OBJ32-NEXT:       Index: 2
; OBJ32-NEXT:       SectionLen: 0
; OBJ32-NEXT:       ParameterHashIndex: 0x0
; OBJ32-NEXT:       TypeChkSectNum: 0x0
; OBJ32-NEXT:       SymbolAlignmentLog2: 0
; OBJ32-NEXT:       SymbolType: XTY_ER (0x0)
; OBJ32-NEXT:       StorageMappingClass: XMC_UA (0x4)
; OBJ32-NEXT:       StabInfoIndex: 0x0
; OBJ32-NEXT:       StabSectNum: 0x0
; OBJ32-NEXT:     }
; OBJ32:          Index: 5
; OBJ32-NEXT:     Name: relro-section
; OBJ32-NEXT:     Value (RelocatableAddress): 0x0
; OBJ32-NEXT:     Section: .data
; OBJ32-NEXT:     Type: 0x0
; OBJ32-NEXT:     StorageClass: C_HIDEXT (0x6B)
; OBJ32-NEXT:     NumberOfAuxEntries: 1
; OBJ32-NEXT:     CSECT Auxiliary Entry {
; OBJ32-NEXT:       Index: 6
; OBJ32-NEXT:       SectionLen: 4
; OBJ32-NEXT:       ParameterHashIndex: 0x0
; OBJ32-NEXT:       TypeChkSectNum: 0x0
; OBJ32-NEXT:       SymbolAlignmentLog2: 2
; OBJ32-NEXT:       SymbolType: XTY_SD (0x1)
; OBJ32-NEXT:       StorageMappingClass: XMC_RW (0x5)
; OBJ32-NEXT:       StabInfoIndex: 0x0
; OBJ32-NEXT:       StabSectNum: 0x0
; OBJ32-NEXT:     }
; OBJ32:       ]

; OBJ64:      Relocations [
; OBJ64-NEXT:   Section (index: 2) .data {
; OBJ64-NEXT:     0x0 R_POS var(1) 0x3F
; OBJ64-NEXT:   }
; OBJ64-NEXT: ]
; OBJ64-NEXT: Symbols [
; OBJ64:          Index: 1
; OBJ64-NEXT:     Name: var
; OBJ64-NEXT:     Value (RelocatableAddress): 0x0
; OBJ64-NEXT:     Section: N_UNDEF
; OBJ64-NEXT:     Type: 0x0
; OBJ64-NEXT:     StorageClass: C_EXT (0x2)
; OBJ64-NEXT:     NumberOfAuxEntries: 1
; OBJ64-NEXT:     CSECT Auxiliary Entry {
; OBJ64-NEXT:       Index: 2
; OBJ64-NEXT:       SectionLen: 0
; OBJ64-NEXT:       ParameterHashIndex: 0x0
; OBJ64-NEXT:       TypeChkSectNum: 0x0
; OBJ64-NEXT:       SymbolAlignmentLog2: 0
; OBJ64-NEXT:       SymbolType: XTY_ER (0x0)
; OBJ64-NEXT:       StorageMappingClass: XMC_UA (0x4)
; OBJ64-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:     }
; OBJ64:          Index: 5
; OBJ64-NEXT:     Name: relro-section
; OBJ64-NEXT:     Value (RelocatableAddress): 0x0
; OBJ64-NEXT:     Section: .data
; OBJ64-NEXT:     Type: 0x0
; OBJ64-NEXT:     StorageClass: C_HIDEXT (0x6B)
; OBJ64-NEXT:     NumberOfAuxEntries: 1
; OBJ64-NEXT:     CSECT Auxiliary Entry {
; OBJ64-NEXT:       Index: 6
; OBJ64-NEXT:       SectionLen: 8
; OBJ64-NEXT:       ParameterHashIndex: 0x0
; OBJ64-NEXT:       TypeChkSectNum: 0x0
; OBJ64-NEXT:       SymbolAlignmentLog2: 3
; OBJ64-NEXT:       SymbolType: XTY_SD (0x1)
; OBJ64-NEXT:       StorageMappingClass: XMC_RW (0x5)
; OBJ64-NEXT:       Auxiliary Type: AUX_CSECT (0xFB)
; OBJ64-NEXT:     }
; OBJ64:      ]
