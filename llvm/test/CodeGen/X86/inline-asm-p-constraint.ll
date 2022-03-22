; RUN: llc -mtriple=x86_64-unknown-unknown -no-integrated-as < %s 2>&1 | FileCheck %s

define i8* @foo(i8* %ptr) {
; CHECK-LABEL: foo:
  %1 = tail call i8* asm "lea $1, $0", "=r,p,~{dirflag},~{fpsr},~{flags}"(i8* %ptr)
; CHECK:      #APP
; CHECK-NEXT: lea (%rdi), %rax
; CHECK-NEXT: #NO_APP
  ret i8* %1
; CHECK-NEXT: retq
}
