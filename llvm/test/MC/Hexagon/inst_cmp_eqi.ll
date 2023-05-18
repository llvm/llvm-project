;; RUN: llc -mtriple=hexagon-unknown-elf -filetype=obj %s -o - \
;; RUN: | llvm-objdump --no-print-imm-hex -d - | FileCheck %s

define i1 @f0(i32 %a0) {
  %v0 = icmp eq i32 %a0, 42
  ret i1 %v0
}

; CHECK: p0 = cmp.eq(r0,#42)
; CHECK: r0 = mux(p0,#1,#0)
; CHECK: jumpr r31
