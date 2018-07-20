; RUN: llc -mtriple x86_64-elf -filetype obj -o - %s | llvm-readobj -elf-linker-options - | FileCheck %s
; REQUIRES: x86-registered-target

!llvm.linker.options = !{!0, !1}

!0 = !{!"option 0", !"value 0"}
!1 = !{!"option 1", !"value 1"}

; CHECK: LinkerOptions [
; CHECK:  option 0: value 0
; CHECK:  option 1: value 1
; CHECK: ]
