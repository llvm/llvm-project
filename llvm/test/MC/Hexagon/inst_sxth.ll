; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
; RUN: | llvm-objdump -d - | FileCheck %s

define i32 @foo (i16 %a)
{
  %1 = sext i16 %a to i32
  ret i32 %1
}

; CHECK: c0 3f 00 54 54003fc0