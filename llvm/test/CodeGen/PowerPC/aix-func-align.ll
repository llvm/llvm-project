; This test tries to verify if a csect containing code would have the correct alignment.

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefixes=SYMS,SYMS32 %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -xcoff-traceback-table=false -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --syms %t64.o | FileCheck --check-prefixes=SYMS,SYMS64 %s

define i32 @foo()  align 32 {
entry:
  ret i32 0
}

define i32 @bar()  align 64 {
entry:
  ret i32 0
}

; CHECK:      .csect [PR],6
; CHECK-NEXT: .foo:

; CHECK:      .csect [PR],6
; CHECK-NEXT: .bar:

; SYMS:       Symbol {{[{][[:space:]] *}}Index: [[#INDX:]]{{[[:space:]] *Name: $}}
; SYMS-NEXT:    Value (RelocatableAddress): 0x0
; SYMS-NEXT:    Section: .text
; SYMS-NEXT:    Type: 0x0
; SYMS-NEXT:    StorageClass: C_HIDEXT (0x6B)
; SYMS-NEXT:    NumberOfAuxEntries: 1
; SYMS-NEXT:    CSECT Auxiliary Entry {
; SYMS-NEXT:      Index: [[#INDX+1]]
; SYMS-NEXT:      SectionLen: 72
; SYMS-NEXT:      ParameterHashIndex: 0x0
; SYMS-NEXT:      TypeChkSectNum: 0x0
; SYMS-NEXT:      SymbolAlignmentLog2: 6
; SYMS-NEXT:      SymbolType: XTY_SD (0x1)
; SYMS-NEXT:      StorageMappingClass: XMC_PR (0x0)
; SYMS32-NEXT:    StabInfoIndex: 0x0
; SYMS32-NEXT:    StabSectNum: 0x0
; SYMS64-NEXT:    Auxiliary Type: AUX_CSECT (0xFB)
; SYMS-NEXT:    }
; SYMS-NEXT:  }
