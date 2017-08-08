;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -d - | FileCheck %s

define i1 @foo (i32 %a)
{
  %1 = icmp eq i32 %a, 42
  ret i1 %1
}

; CHECK: p0 = cmp.eq(r0,#42)
; CHECK: r0 = p0
; CHECK: jumpr r31
