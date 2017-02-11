;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -d - | FileCheck %s

define i1 @foo (i32 %a, i32 %b)
{
  %1 = icmp slt i32 %a, %b
  ret i1 %1
}

; CHECK: p0 = cmp.gt(r1,r0)
; CHECK: r0 = p0
; CHECK: jumpr r31
