;; Test to make sure a symbol name that starts with an 'L' could be succesfully
;; consumed. Note that this could fail if PrivateGlobalPrefix returns
;; 'L'/'.L' instead of 'L..' because the resulting symbol gets created as
;; a temporary symbol.

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=XCOFF,XCOFF32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=XCOFF,XCOFF64 %s

@La = external global i32, align 4

declare i32 @Lb(...)

define void @foo() {
entry:
  store i32 1, ptr @La, align 4
  call i32 @Lb()
  ret void
}

; XCOFF:         Index: [[#IND:]]{{.*}}{{[[:space:]] *}}Name: .Lb
; XCOFF-NEXT:    Value (RelocatableAddress): 0x0
; XCOFF-NEXT:    Section: N_UNDEF
; XCOFF-NEXT:    Type: 0x0
; XCOFF-NEXT:    StorageClass: C_EXT (0x2)
; XCOFF-NEXT:    NumberOfAuxEntries: 1
; XCOFF-NEXT:    CSECT Auxiliary Entry {
; XCOFF-NEXT:      Index: [[#IND+1]]
; XCOFF-NEXT:      SectionLen: 0
; XCOFF-NEXT:      ParameterHashIndex: 0x0
; XCOFF-NEXT:      TypeChkSectNum: 0x0
; XCOFF-NEXT:      SymbolAlignmentLog2: 0
; XCOFF-NEXT:      SymbolType: XTY_ER (0x0)
; XCOFF-NEXT:      StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:    StabInfoIndex: 0x0
; XCOFF32-NEXT:    StabSectNum: 0x0
; XCOFF64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; XCOFF-NEXT:    }
; XCOFF-NEXT:  }
; XCOFF-NEXT:  Symbol {
; XCOFF-NEXT:    Index: [[#IND+2]]
; XCOFF-NEXT:    Name: La
; XCOFF-NEXT:    Value (RelocatableAddress): 0x0
; XCOFF-NEXT:    Section: N_UNDEF
; XCOFF-NEXT:    Type: 0x0
; XCOFF-NEXT:    StorageClass: C_EXT (0x2)
; XCOFF-NEXT:    NumberOfAuxEntries: 1
; XCOFF-NEXT:    CSECT Auxiliary Entry {
; XCOFF-NEXT:      Index: [[#IND+3]]
; XCOFF-NEXT:      SectionLen: 0
; XCOFF-NEXT:      ParameterHashIndex: 0x0
; XCOFF-NEXT:      TypeChkSectNum: 0x0
; XCOFF-NEXT:      SymbolAlignmentLog2: 0
; XCOFF-NEXT:      SymbolType: XTY_ER (0x0)
; XCOFF-NEXT:      StorageMappingClass: XMC_UA (0x4)
; XCOFF32-NEXT:    StabInfoIndex: 0x0
; XCOFF32-NEXT:    StabSectNum: 0x0
; XCOFF64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; XCOFF-NEXT:    }
; XCOFF-NEXT:  }
