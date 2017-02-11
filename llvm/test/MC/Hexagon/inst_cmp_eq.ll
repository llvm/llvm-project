;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -d - | FileCheck %s

define i1 @foo (i32 %a, i32 %b)
{
  %1 = icmp eq i32 %a, %b
  ret i1 %1
}

; CHECK: p0 = cmp.eq(r0,r1)
; CHECK: r0 = p0
; CHECK: jumpr r31
