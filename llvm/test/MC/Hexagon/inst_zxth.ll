; RUN: llc -mtriple=hexagon -filetype=obj %s -o - \
; RUN: | llvm-objdump -d - | FileCheck %s

define i32 @foo (i16 %a)
{
  %1 = zext i16 %a to i32
  ret i32 %1
}

; CHECK: c0 3f 00 56 56003fc0
