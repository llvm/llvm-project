; Test indirect calls.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; We must allocate 160 bytes for the callee and save and restore %r14.
define i64 @f1(ptr %bar) {
; CHECK-LABEL: f1:
; CHECK: stmg %r14, %r15, 112(%r15)
; CHECK: aghi %r15, -160
; CHECK: basr %r14, %r2
; CHECK: lmg %r14, %r15, 272(%r15)
; CHECK: br %r14
  %ret = call i64 %bar()
  %inc = add i64 %ret, 1
  ret i64 %inc
}
