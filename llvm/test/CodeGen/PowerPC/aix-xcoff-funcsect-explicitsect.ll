; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -filetype=obj -function-sections -o %t.o < %s
; RUN: llvm-readobj -s %t.o | FileCheck %s

define dso_local signext i32 @foo1() section "sect" {
entry:
  ret i32 1
}

define dso_local signext i32 @foo2() section "sect2" {
entry:
  ret i32 2
}

define dso_local signext i32 @foo3()  section "sect2" {
entry:
  ret i32 3
}

define dso_local signext i32 @foo4() {
entry:
  ret i32 4
}

; CHECK:       Symbol {{[{][[:space:]] *}}Index: [[#INDX:]]{{[[:space:]] *}}Name: sect
; CHECK-NEXT:    Value (RelocatableAddress): 0x0
; CHECK-NEXT:    Section: .text
; CHECK-NEXT:    Type: 0x0
; CHECK-NEXT:    StorageClass: C_HIDEXT (0x6B)
; CHECK-NEXT:    NumberOfAuxEntries: 1
; CHECK-NEXT:    CSECT Auxiliary Entry {
; CHECK-NEXT:      Index: [[#INDX+1]]
; CHECK-NEXT:      SectionLen: 8
; CHECK-NEXT:      ParameterHashIndex: 0x0
; CHECK-NEXT:      TypeChkSectNum: 0x0
; CHECK-NEXT:      SymbolAlignmentLog2: 5
; CHECK-NEXT:      SymbolType: XTY_SD (0x1)
; CHECK-NEXT:      StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:      StabInfoIndex: 0x0
; CHECK-NEXT:      StabSectNum: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:  }
; CHECK-NEXT:  Symbol {
; CHECK-NEXT:    Index: [[#INDX+2]]
; CHECK-NEXT:    Name: .foo1
; CHECK-NEXT:    Value (RelocatableAddress): 0x0
; CHECK-NEXT:    Section: .text
; CHECK-NEXT:    Type: 0x0
; CHECK-NEXT:    StorageClass: C_EXT (0x2)
; CHECK-NEXT:    NumberOfAuxEntries: 1
; CHECK-NEXT:    CSECT Auxiliary Entry {
; CHECK-NEXT:      Index:  [[#INDX+3]]
; CHECK-NEXT:      ContainingCsectSymbolIndex: [[#INDX]]
; CHECK-NEXT:      ParameterHashIndex: 0x0
; CHECK-NEXT:      TypeChkSectNum: 0x0
; CHECK-NEXT:      SymbolAlignmentLog2: 0
; CHECK-NEXT:      SymbolType: XTY_LD (0x2)
; CHECK-NEXT:      StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:      StabInfoIndex: 0x0
; CHECK-NEXT:      StabSectNum: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:  }
; CHECK-NEXT:  Symbol {
; CHECK-NEXT:    Index: [[#INDX+4]]
; CHECK-NEXT:    Name: sect2
; CHECK-NEXT:    Value (RelocatableAddress): 0x20
; CHECK-NEXT:    Section: .text
; CHECK-NEXT:    Type: 0x0
; CHECK-NEXT:    StorageClass: C_HIDEXT (0x6B)
; CHECK-NEXT:    NumberOfAuxEntries: 1
; CHECK-NEXT:    CSECT Auxiliary Entry {
; CHECK-NEXT:      Index: [[#INDX+5]]
; CHECK-NEXT:      SectionLen: 24
; CHECK-NEXT:      ParameterHashIndex: 0x0
; CHECK-NEXT:      TypeChkSectNum: 0x0
; CHECK-NEXT:      SymbolAlignmentLog2: 5
; CHECK-NEXT:      SymbolType: XTY_SD (0x1)
; CHECK-NEXT:      StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:      StabInfoIndex: 0x0
; CHECK-NEXT:      StabSectNum: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:  }
; CHECK-NEXT:  Symbol {
; CHECK-NEXT:    Index: [[#INDX+6]]
; CHECK-NEXT:    Name: .foo2
; CHECK-NEXT:    Value (RelocatableAddress): 0x20
; CHECK-NEXT:    Section: .text
; CHECK-NEXT:    Type: 0x0
; CHECK-NEXT:    StorageClass: C_EXT (0x2)
; CHECK-NEXT:    NumberOfAuxEntries: 1
; CHECK-NEXT:    CSECT Auxiliary Entry {
; CHECK-NEXT:      Index: [[#INDX+7]]
; CHECK-NEXT:      ContainingCsectSymbolIndex: [[#INDX+4]]
; CHECK-NEXT:      ParameterHashIndex: 0x0
; CHECK-NEXT:      TypeChkSectNum: 0x0
; CHECK-NEXT:      SymbolAlignmentLog2: 0
; CHECK-NEXT:      SymbolType: XTY_LD (0x2)
; CHECK-NEXT:      StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:      StabInfoIndex: 0x0
; CHECK-NEXT:      StabSectNum: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:  }
; CHECK-NEXT:  Symbol {
; CHECK-NEXT:    Index: [[#INDX+8]]
; CHECK-NEXT:    Name: .foo3
; CHECK-NEXT:    Value (RelocatableAddress): 0x30
; CHECK-NEXT:    Section: .text
; CHECK-NEXT:    Type: 0x0
; CHECK-NEXT:    StorageClass: C_EXT (0x2)
; CHECK-NEXT:    NumberOfAuxEntries: 1
; CHECK-NEXT:    CSECT Auxiliary Entry {
; CHECK-NEXT:      Index: [[#INDX+9]]
; CHECK-NEXT:      ContainingCsectSymbolIndex: [[#INDX+4]]
; CHECK-NEXT:      ParameterHashIndex: 0x0
; CHECK-NEXT:      TypeChkSectNum: 0x0
; CHECK-NEXT:      SymbolAlignmentLog2: 0
; CHECK-NEXT:      SymbolType: XTY_LD (0x2)
; CHECK-NEXT:      StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:      StabInfoIndex: 0x0
; CHECK-NEXT:      StabSectNum: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:  }
; CHECK-NEXT:  Symbol {
; CHECK-NEXT:    Index: [[#INDX+10]]
; CHECK-NEXT:    Name: .foo4
; CHECK-NEXT:    Value (RelocatableAddress): 0x40
; CHECK-NEXT:    Section: .text
; CHECK-NEXT:    Type: 0x0
; CHECK-NEXT:    StorageClass: C_EXT (0x2)
; CHECK-NEXT:    NumberOfAuxEntries: 1
; CHECK-NEXT:    CSECT Auxiliary Entry {
; CHECK-NEXT:      Index: 16
; CHECK-NEXT:      SectionLen: 8
; CHECK-NEXT:      ParameterHashIndex: 0x0
; CHECK-NEXT:      TypeChkSectNum: 0x0
; CHECK-NEXT:      SymbolAlignmentLog2: 5
; CHECK-NEXT:      SymbolType: XTY_SD (0x1)
; CHECK-NEXT:      StorageMappingClass: XMC_PR (0x0)
; CHECK-NEXT:      StabInfoIndex: 0x0
; CHECK-NEXT:      StabSectNum: 0x0
; CHECK-NEXT:    }
; CHECK-NEXT:  }
