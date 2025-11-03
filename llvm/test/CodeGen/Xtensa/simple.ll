; RUN: llc -mtriple=xtensa -filetype=asm %s -o - | FileCheck %s
; RUN: llc -mtriple=xtensa -filetype=obj %s -o - | llvm-objdump --arch=xtensa  -d - | FileCheck %s --check-prefix=DUMP

; CHECK:  .text
; DUMP:   file format elf32-xtensa
define void @f() {
  ret void
}
