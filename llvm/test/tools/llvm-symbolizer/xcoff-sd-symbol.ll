;; FIXME: use assembly rather than LLVM IR once integrated assembler supports
;; AIX assembly syntax.

; REQUIRES: powerpc-registered-target
; RUN: llc -filetype=obj -o %t -mtriple=powerpc-aix-ibm-xcoff -function-sections < %s
; RUN: llvm-symbolizer --obj=%t 'CODE 0x0' 'CODE 0x20' | \
; RUN:   FileCheck %s

define void @foo() {
entry:
  ret void
}

define void @foo1() {
entry:
  ret void
}

; CHECK: ??
; CHECK: ??:0:0
; CHECK-EMPTY:

; CHECK: ??
; CHECK: ??:0:0
; CHECK-EMPTY:
