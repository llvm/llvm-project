; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck --check-prefixes=XCOFF,XCOFF32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck --check-prefixes=XCOFF,XCOFF64 %s

@_ZN3Foo1aE = available_externally constant i32 -1

; CHECK: .extern  _ZN3Foo1aE[UA]

; XCOFF:       Index: [[#Index:]]{{.*}}{{[[:space:]] *}}Name: _ZN3Foo1aE
; XCOFF-NEXT:  Value (RelocatableAddress): 0x0
; XCOFF-NEXT:  Section: N_UNDEF
; XCOFF-NEXT:  Type: 0x0
; XCOFF-NEXT:  StorageClass: C_EXT (0x2)
; XCOFF-NEXT:  NumberOfAuxEntries: 1
; XCOFF-NEXT:  CSECT Auxiliary Entry {
; XCOFF-NEXT:    Index: [[#Index+1]]
; XCOFF-NEXT:    SectionLen: 0
; XCOFF-NEXT:    ParameterHashIndex: 0x0
; XCOFF-NEXT:    TypeChkSectNum: 0x0
; XCOFF-NEXT:    SymbolAlignmentLog2: 0
; XCOFF-NEXT:    SymbolType: XTY_ER (0x0)
; XCOFF-NEXT:    StorageMappingClass: XMC_UA (0x4)
; XCOFF32:       StabInfoIndex: 0x0
; XCOFF32-NEXT:  StabSectNum: 0x0
; XCOFF64:       Auxiliary Type: AUX_CSECT (0xFB)
