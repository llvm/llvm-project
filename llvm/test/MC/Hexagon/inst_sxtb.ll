; RUN: llc -mtriple=hexagon -filetype=obj %s -o - \
; RUN: | llvm-objdump -d - | FileCheck %s

define i32 @foo (i8 %a)
{
  %1 = sext i8 %a to i32
  ret i32 %1
}

; CHECK: c0 3f 00 55 55003fc0
