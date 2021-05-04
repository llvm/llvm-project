;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump -d - | FileCheck %s

define i1 @f0(i32 %a0, i32 %a1) {
  %v0 = icmp ult i32 %a0, %a1
  ret i1 %v0
}

; CHECK: p0 = cmp.gtu(r1,r0)
; CHECK: r0 = mux(p0,#1,#0)
; CHECK: jumpr r31
