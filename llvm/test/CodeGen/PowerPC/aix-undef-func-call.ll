; RUN: llc -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj  --symbols %t.o | FileCheck %s

define void @bar() {
entry:
  call void bitcast (void (...)* @foo to void ()*)()
  ret void
}

declare void @foo(...)

;CHECK: Symbol {
;CHECK:   Name: .foo
;CHECK-NEXT:   Value (RelocatableAddress): 0x0
;CHECK-NEXT:   Section: N_UNDEF
;CHECK-NEXT:   Type: 0x0
;CHECK-NEXT:   StorageClass: C_EXT (0x2)
;CHECK-NEXT:   NumberOfAuxEntries: 1
;CHECK-NEXT:   CSECT Auxiliary Entry {
;CHECK:          SectionLen: 0
;CHECK-NEXT:     ParameterHashIndex: 0x0
;CHECK-NEXT:     TypeChkSectNum: 0x0
;CHECK-NEXT:     SymbolAlignmentLog2: 0
;CHECK-NEXT:     SymbolType: XTY_ER (0x0)
;CHECK-NEXT:     StorageMappingClass: XMC_PR (0x0)
;CHECK-NEXT:     StabInfoIndex: 0x0
;CHECK-NEXT:     StabSectNum: 0x0
;CHECK-NEXT:   }
;CHECK-NEXT: }
