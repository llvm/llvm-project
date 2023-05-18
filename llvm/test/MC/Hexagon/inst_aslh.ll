;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i32 @foo (i32 %a)
{
  %1 = shl i32 %a, 16
  ret i32 %1
}

; CHECK:   0000 00400070 00c09f52
