; RUN: llc -mtriple=x86_64-pc-windows-msvc -tailcallopt %s -o - | FileCheck %s

; When RAX is live into the entry block (here the first argument of a regcall
; function), the probed allocations of a tail call reserve and of the main frame
; both have to preserve it: RAX is pushed, the probe is given the size of the
; rest of the allocation, and RAX is reloaded from the slot that the push
; created.

declare x86_regcallcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

define x86_regcallcc void @f(i64 %a, i64 %b) "stack-probe-size"="32" {
; CHECK-LABEL: f:
; The reserve is 16 bytes: the 8 pushed here plus the 8 probed.
; CHECK:         pushq %rax
; CHECK-NEXT:    movl $8, %eax
; CHECK-NEXT:    callq __chkstk
; CHECK-NEXT:    subq %rax, %rsp
; CHECK-NEXT:    movq 8(%rsp), %rax
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK:         pushq %rax
; CHECK-NEXT:    movl $24, %eax
; CHECK-NEXT:    callq __chkstk
; CHECK-NEXT:    subq %rax, %rsp
; CHECK-NEXT:    movq 24(%rsp), %rax
; CHECK-NEXT:    .seh_stackalloc 32
; CHECK:         .seh_endprologue
  tail call x86_regcallcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
}
