;; RUN: llc -mtriple=hexagon-unknown-elf -disable-hsdr -filetype=obj %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i64 @foo (i64 %a, i64 %b)
{
  %1 = xor i64 %a, %b
  ret i64 %1
}

; CHECK:  0000 8042e0d3 00c09f52
