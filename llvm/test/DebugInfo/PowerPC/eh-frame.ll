; RUN: llc -filetype=obj -mtriple=powerpc %s -o %t32.o
; RUN: llvm-readobj -r %t32.o | FileCheck %s --check-prefix=PPC_REL
; RUN: llvm-dwarfdump --eh-frame %t32.o 2>&1 | FileCheck %s --check-prefix=PPC

; PPC_REL:      R_PPC_REL32 .text 0x0
; PPC_REL-NEXT: R_PPC_REL32 .text 0x4

; PPC-NOT: warning:
; PPC: FDE cie=00000000 pc=00000000...00000004
; PPC: FDE cie=00000000 pc=00000004...00000008

; RUN: llc -filetype=obj -mtriple=ppc64 %s -o %t64.o
; RUN: llvm-readobj -r %t64.o | FileCheck %s --check-prefix=PPC64_REL
; RUN: llvm-dwarfdump --eh-frame %t64.o 2>&1 | FileCheck %s --check-prefix=PPC64

; PPC64_REL:      R_PPC64_REL32 .text 0x0
; PPC64_REL-NEXT: R_PPC64_REL32 .text 0x10

; PPC64-NOT: warning:
; PPC64: FDE cie=00000000 pc=00000000...00000010
; PPC64: FDE cie=00000000 pc=00000010...00000020

; RUN: llc -filetype=obj -mtriple=ppc64le -code-model=large %s -o %t64l.o
; RUN: llvm-readobj -r %t64l.o | FileCheck %s --check-prefix=PPC64L_REL
; RUN: llvm-dwarfdump --eh-frame %t64l.o 2>&1 | FileCheck %s --check-prefix=PPC64

; PPC64L_REL:      R_PPC64_REL64 .text 0x0
; PPC64L_REL-NEXT: R_PPC64_REL64 .text 0x10

define void @foo() {
entry:
  ret void
}

define void @bar() {
entry:
  ret void
}
