; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj  --symbols %t.o | FileCheck %s

define void @foo() {
entry:
  ret void
}

; CHECK:      File: {{.*}}aix-func-dsc-gen.ll.tmp.o
; CHECK-NEXT: Format: aixcoff-rs6000
; CHECK-NEXT: Arch: powerpc
; CHECK-NEXT: AddressSize: 32bit
; CHECK:        Symbol {
; CHECK-NEXT:     Index: 0
; CHECK-NEXT:     Name: <stdin>
; CHECK-NEXT:     Value (SymbolTableIndex): 0x0
; CHECK-NEXT:     Section: N_DEBUG
; CHECK-NEXT:     Source Language ID: TB_CPLUSPLUS (0x9)
; CHECK-NEXT:     CPU Version ID: TCPU_COM (0x3)
; CHECK-NEXT:     StorageClass: C_FILE (0x67)
; CHECK-NEXT:     NumberOfAuxEntries: 0
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Index: [[#Index:]]
; CHECK-NEXT:     Name:
; CHECK-NEXT:     Value (RelocatableAddress): 0x0
; CHECK-NEXT:     Section: .text
; CHECK-NEXT:     Type: 0x0
; CHECK-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECK-NEXT:     NumberOfAuxEntries: 1
; CHECK-NEXT:     CSECT Auxiliary Entry {
; CHECK-NEXT:       Index: [[#Index+1]]
; CHECK-NEXT:       SectionLen: 4
; CHECK-NEXT:       ParameterHashIndex: 0x0
; CHECK-NEXT:       TypeChkSectNum: 0x0
; CHECK-NEXT:       SymbolAlignmentLog2: 5
; CHECK-NEXT:       SymbolType: XTY_SD (0x1)
; CHECK-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:       StabInfoIndex: 0x0
; CHECK-NEXT:       StabSectNum: 0x0
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Index: [[#Index+2]]
; CHECK-NEXT:     Name: .foo
; CHECK-NEXT:     Value (RelocatableAddress): 0x0
; CHECK-NEXT:     Section: .text
; CHECK-NEXT:     Type: 0x0
; CHECK-NEXT:     StorageClass: C_EXT (0x2)
; CHECK-NEXT:     NumberOfAuxEntries: 1
; CHECK-NEXT:     CSECT Auxiliary Entry {
; CHECK-NEXT:       Index: [[#Index+3]]
; CHECK-NEXT:       ContainingCsectSymbolIndex: [[#Index]]
; CHECK-NEXT:       ParameterHashIndex: 0x0
; CHECK-NEXT:       TypeChkSectNum: 0x0
; CHECK-NEXT:       SymbolAlignmentLog2: 0
; CHECK-NEXT:       SymbolType: XTY_LD (0x2)
; CHECK-NEXT:       StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:       StabInfoIndex: 0x0
; CHECK-NEXT:       StabSectNum: 0x0
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Index: [[#Index+4]]
; CHECK-NEXT:     Name: foo
; CHECK-NEXT:     Value (RelocatableAddress): 0x4
; CHECK-NEXT:     Section: .data
; CHECK-NEXT:     Type: 0x0
; CHECK-NEXT:     StorageClass: C_EXT (0x2)
; CHECK-NEXT:     NumberOfAuxEntries: 1
; CHECK-NEXT:     CSECT Auxiliary Entry {
; CHECK-NEXT:       Index: [[#Index+5]]
; CHECK-NEXT:       SectionLen: 12
; CHECK-NEXT:       ParameterHashIndex: 0x0
; CHECK-NEXT:       TypeChkSectNum: 0x0
; CHECK-NEXT:       SymbolAlignmentLog2: 2
; CHECK-NEXT:       SymbolType: XTY_SD (0x1)
; CHECK-NEXT:       StorageMappingClass: XMC_DS (0xA)
; CHECK-NEXT:       StabInfoIndex: 0x0
; CHECK-NEXT:       StabSectNum: 0x0
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   Symbol {
; CHECK-NEXT:     Index: [[#Index+6]]
; CHECK-NEXT:     Name: TOC
; CHECK-NEXT:     Value (RelocatableAddress): 0x10
; CHECK-NEXT:     Section: .data
; CHECK-NEXT:     Type: 0x0
; CHECK-NEXT:     StorageClass: C_HIDEXT (0x6B)
; CHECK-NEXT:     NumberOfAuxEntries: 1
; CHECK-NEXT:     CSECT Auxiliary Entry {
; CHECK-NEXT:       Index: [[#Index+7]]
; CHECK-NEXT:       SectionLen: 0
; CHECK-NEXT:       ParameterHashIndex: 0x0
; CHECK-NEXT:       TypeChkSectNum: 0x0
; CHECK-NEXT:       SymbolAlignmentLog2: 2
; CHECK-NEXT:       SymbolType: XTY_SD (0x1)
; CHECK-NEXT:       StorageMappingClass: XMC_TC0 (0xF)
; CHECK-NEXT:       StabInfoIndex: 0x0
; CHECK-NEXT:       StabSectNum: 0x0
; CHECK-NEXT:     }
; CHECK-NEXT:   }
