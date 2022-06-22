; RUN: llc -mtriple=x86_64-unknown-unknown -no-integrated-as < %s 2>&1 | FileCheck %s

define ptr @foo(ptr %ptr) {
; CHECK-LABEL: foo:
  %1 = tail call ptr asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(ptr %ptr)
; CHECK:      #APP
; CHECK-NEXT: lea (%rdi), %rax
; CHECK-NEXT: #NO_APP
  ret ptr %1
; CHECK-NEXT: retq
}
