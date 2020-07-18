; RUN: llc -filetype=obj -mtriple=aarch64 %s -o %t.o
; RUN: llvm-readobj -r %t.o | FileCheck %s --check-prefix=REL32
; RUN: llvm-dwarfdump --eh-frame %t.o 2>&1 | FileCheck %s

; REL32:      R_AARCH64_PREL32 .text 0x0
; REL32-NEXT: R_AARCH64_PREL32 .text 0x4

; CHECK-NOT:  warning:
; CHECK: FDE cie=00000000 pc=00000000...00000004
;; TODO Take relocation into consideration
; CHECK: FDE cie=00000000 pc=00000000...00000004

define void @foo() {
entry:
  ret void
}

define void @bar() {
entry:
  ret void
}
