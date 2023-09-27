; RUN: llc -mtriple=hexagon -filetype=obj %s -o - \
; RUN: | llvm-objdump -d - | FileCheck %s

define i32 @foo (i32 %a, i32 %b)
{
  %1 = add i32 %a, %b
  ret i32 %1
}

; CHECK:  c0 3f 10 58 58103fc0
