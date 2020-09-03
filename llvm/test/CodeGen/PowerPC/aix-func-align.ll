; This test tries to verify if a csect containing code would have the correct alignment.

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=SYMS %s

; RUN: not --crash llc -mtriple powerpc64-ibm-aix-xcoff -filetype=obj < %s 2>&1 | \
; RUN:   FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

define i32 @foo()  align 32 {
entry:
  ret i32 0
}

define i32 @bar()  align 64 {
entry:
  ret i32 0
}

; CHECK:      .csect .text[PR],6
; CHECK-NEXT: .foo:

; CHECK:      .csect .text[PR],6
; CHECK-NEXT: .bar:

; SYMS:       Symbol {{[{][[:space:]] *}}Index: [[#INDX:]]{{[[:space:]] *}}Name: .text
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
; SYMS-NEXT:      StabInfoIndex: 0x0
; SYMS-NEXT:      StabSectNum: 0x0
; SYMS-NEXT:    }
; SYMS-NEXT:  }
