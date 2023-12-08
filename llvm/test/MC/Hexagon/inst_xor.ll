;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i32 @foo (i32 %a, i32 %b)
{
  %1 = xor i32 %a, %b
  ret i32 %1
}

; CHECK:   0000 004160f1 00c09f52
