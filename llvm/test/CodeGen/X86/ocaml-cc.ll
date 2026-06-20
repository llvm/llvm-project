; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

; The OCaml calling convention (cc128) passes integer arguments in the OCaml
; register order starting with R14, R15, RAX, RBX, ... and saves no
; callee-saved registers (CSR_64_OCaml).

; CHECK-LABEL: add:
; First two i64 args are R14 and R15.
; CHECK: addq %r15, %r14
; CHECK: retq
define cc128 i64 @add(i64 %a, i64 %b) {
  %r = add i64 %a, %b
  ret i64 %r
}

; CHECK-LABEL: caller:
; OCaml functions do not preserve callee-saved registers across calls.
; CHECK-NOT: pushq %rbx
; CHECK-NOT: pushq %r12
; CHECK-NOT: pushq %r13
; CHECK-NOT: pushq %rbp
; CHECK: callq add
; CHECK: retq
define cc128 i64 @caller(i64 %x) {
  %r = call cc128 i64 @add(i64 %x, i64 %x)
  ret i64 %r
}
