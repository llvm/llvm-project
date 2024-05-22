; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t.o < %s
; RUN: llvm-readobj --symbols %t.o | FileCheck -D#NFA=2 --check-prefix=OBJ %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -filetype=obj -o %t64.o < %s
; RUN: llvm-readobj --symbols %t64.o | FileCheck -D#NFA=2 --check-prefix=OBJ %s

define available_externally i32 @foo(i32 %a) {
entry:
  ret i32 %a
}

; CHECK: .extern .foo[PR]
; CHECK: .extern foo[DS]

; OBJ:      Name: .foo
; OBJ-NEXT: Value (RelocatableAddress): 0x0
; OBJ-NEXT: Section: N_UNDEF
; OBJ-NEXT: Type: 0x0
; OBJ-NEXT: StorageClass: C_EXT (0x2)
; OBJ-NEXT: NumberOfAuxEntries: 1
; OBJ-NEXT: CSECT Auxiliary Entry {
; OBJ-NEXT:   Index: [[#NFA+2]]
; OBJ-NEXT:   SectionLen: 0
; OBJ-NEXT:   ParameterHashIndex: 0x0
; OBJ-NEXT:   TypeChkSectNum: 0x0
; OBJ-NEXT:   SymbolAlignmentLog2: 0
; OBJ-NEXT:   SymbolType: XTY_ER (0x0)
; OBJ-NEXT:   StorageMappingClass: XMC_PR (0x0)

; OBJ:      Name: foo
; OBJ-NEXT: Value (RelocatableAddress): 0x0
; OBJ-NEXT: Section: N_UNDEF
; OBJ-NEXT: Type: 0x0
; OBJ-NEXT: StorageClass: C_EXT (0x2)
; OBJ-NEXT: NumberOfAuxEntries: 1
; OBJ-NEXT: CSECT Auxiliary Entry {
; OBJ-NEXT:   Index: [[#NFA+4]]
; OBJ-NEXT:   SectionLen: 0
; OBJ-NEXT:   ParameterHashIndex: 0x0
; OBJ-NEXT:   TypeChkSectNum: 0x0
; OBJ-NEXT:   SymbolAlignmentLog2: 0
; OBJ-NEXT:   SymbolType: XTY_ER (0x0)
; OBJ-NEXT:   StorageMappingClass: XMC_DS (0xA)
