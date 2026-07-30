;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj -disable-hsdr %s -o - \
;; RUN: | llvm-objdump -s - | FileCheck %s

define i64 @foo (i64 %a, i64 %b)
{
  %1 = or i64 %a, %b
  ret i64 %1
}

; CHECK:  0000 4042e0d3 00c09f52
