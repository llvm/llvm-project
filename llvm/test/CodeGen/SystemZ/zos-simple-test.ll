; A simple, barebones test to check whether assembly can be emitted
; for the z/OS target
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

@a = global i32 0, align 4

define signext i32 @main() {
; CHECK: stdin#C CSECT
; CHECK: C_CODE64 CATTR ALIGN(3),EXECUTABLE,READONLY,RMODE(64)
; CHECK: main:
; CHECK: stdin#C CSECT
; CHECK: C_WSA64 CATTR ALIGN(4),DEFLOAD,NOTEXECUTABLE,RMODE(64),PART(a)
; CHECK: a:
entry:
  ret i32 0
}
