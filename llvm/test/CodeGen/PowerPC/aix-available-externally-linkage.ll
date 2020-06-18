; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | \
; RUN:   FileCheck --check-prefix=XCOFF32 %s

; RUN: not --crash llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN:     -mcpu=pwr4 -mattr=-altivec -filetype=obj -o %t.o 2>&1 < %s | \
; RUN:   FileCheck --check-prefix=XCOFF64 %s
; XCOFF64: LLVM ERROR: 64-bit XCOFF object files are not supported yet.

@_ZN3Foo1aE = available_externally constant i32 -1

; CHECK: .extern  _ZN3Foo1aE[UA]

; XCOFF32:          Index: [[#Index:]]{{.*}}{{[[:space:]] *}}Name: _ZN3Foo1aE
; XCOFF32-NEXT:     Value (RelocatableAddress): 0x0
; XCOFF32-NEXT:     Section: N_UNDEF
; XCOFF32-NEXT:     Type: 0x0
; XCOFF32-NEXT:     StorageClass: C_EXT (0x2)
; XCOFF32-NEXT:     NumberOfAuxEntries: 1
; XCOFF32-NEXT:     CSECT Auxiliary Entry {
; XCOFF32-NEXT:       Index: [[#Index+1]]
; XCOFF32-NEXT:       SectionLen: 0
; XCOFF32-NEXT:       ParameterHashIndex: 0x0
; XCOFF32-NEXT:       TypeChkSectNum: 0x0
; XCOFF32-NEXT:       SymbolAlignmentLog2: 0
; XCOFF32-NEXT:       SymbolType: XTY_ER (0x0)
; XCOFF32-NEXT:       StorageMappingClass: XMC_UA (0x4)
; XCOFF32-NEXT:       StabInfoIndex: 0x0
; XCOFF32-NEXT:       StabSectNum: 0x0
; XCOFF32-NEXT:     }
