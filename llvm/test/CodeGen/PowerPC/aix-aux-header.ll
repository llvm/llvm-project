; RUN: llc -filetype=obj -mtriple=powerpc-ibm-aix-xcoff %s -o - | \
; RUN:   llvm-readobj --auxiliary-header - | FileCheck %s

; CHECK:      AuxiliaryHeader {
; CHECK-NEXT:   Magic: 0x0
; CHECK-NEXT:   Version: 0x2
; CHECK-NEXT:   Size of .text section: 0x24
; CHECK-NEXT:   Size of .data section: 0x14
; CHECK-NEXT:   Size of .bss section: 0x0
; CHECK-NEXT:   Entry point address: 0x0
; CHECK-NEXT:   .text section start address: 0x0
; CHECK-NEXT:   .data section start address: 0x24
; CHECK-NEXT: }

@var = hidden global i32 0, align 4

define hidden i32 @fun() {
entry:
  %0 = load i32, i32* @var, align 4
  ret i32 %0
}
