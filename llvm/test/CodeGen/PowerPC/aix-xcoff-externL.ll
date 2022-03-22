;; Test to make sure a symbol name that starts with an 'L' could be succesfully
;; consumed. Note that this could fail if PrivateGlobalPrefix returns
;; 'L'/'.L' instead of 'L..' because the resulting symbol gets created as
;; a temporary symbol.

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | \
; RUN:   FileCheck --check-prefix=XCOFF32 %s

;; FIXME: currently only fileHeader and sectionHeaders are supported in XCOFF64.

@La = external global i32, align 4

declare i32 @Lb(...)

define void @foo() {
entry:
  store i32 1, i32* @La, align 4
  call i32 bitcast (i32 (...)* @Lb to i32 ()*)()
  ret void
}

; XCOFF32:         Index: [[#IND:]]{{.*}}{{[[:space:]] *}}Name: .Lb
; XCOFF32-NEXT:    Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:    Section: N_UNDEF
; XCOFF32-NEXT:    Type: 0x0
; XCOFF32-NEXT:    StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:    NumberOfAuxEntries: 1
; XCOFF32-NEXT:    CSECT Auxiliary Entry {
; XCOFF32-NEXT:      Index: [[#IND+1]]
; XCOFF32-NEXT:      SectionLen: 0
; XCOFF32-NEXT:      ParameterHashIndex: 0x0
; XCOFF32-NEXT:      TypeChkSectNum: 0x0
; XCOFF32-NEXT:      SymbolAlignmentLog2: 0
; XCOFF32-NEXT:      SymbolType: XTY_ER (0x0)
; XCOFF32-NEXT:      StorageMappingClass: XMC_PR (0x0)
; XCOFF32-NEXT:      StabInfoIndex: 0x0
; XCOFF32-NEXT:      StabSectNum: 0x0
; XCOFF32-NEXT:    }
; XCOFF32-NEXT:  }
; XCOFF32-NEXT:  Symbol {
; XCOFF32-NEXT:    Index: [[#IND+2]]
; XCOFF32-NEXT:    Name: La
; XCOFF32-NEXT:    Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:    Section: N_UNDEF
; XCOFF32-NEXT:    Type: 0x0
; XCOFF32-NEXT:    StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:    NumberOfAuxEntries: 1
; XCOFF32-NEXT:    CSECT Auxiliary Entry {
; XCOFF32-NEXT:      Index: [[#IND+3]]
; XCOFF32-NEXT:      SectionLen: 0
; XCOFF32-NEXT:      ParameterHashIndex: 0x0
; XCOFF32-NEXT:      TypeChkSectNum: 0x0
; XCOFF32-NEXT:      SymbolAlignmentLog2: 0
; XCOFF32-NEXT:      SymbolType: XTY_ER (0x0)
; XCOFF32-NEXT:      StorageMappingClass: XMC_UA (0x4)
; XCOFF32-NEXT:      StabInfoIndex: 0x0
; XCOFF32-NEXT:      StabSectNum: 0x0
; XCOFF32-NEXT:    }
; XCOFF32-NEXT:  }
