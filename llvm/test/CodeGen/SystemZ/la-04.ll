; Test blockaddress.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Do some arbitrary work and return the address of the following label.
define ptr@f1(ptr %addr) {
; CHECK-LABEL: f1:
; CHECK: mvi 0(%r2), 1
; CHECK: [[LABEL:\.L.*]]:
; CHECK: larl %r2, [[LABEL]]
; CHECK: br %r14
entry:
  store i8 1, ptr %addr
  br label %b.lab

b.lab:
  ret ptr blockaddress(@f1, %b.lab)
}
