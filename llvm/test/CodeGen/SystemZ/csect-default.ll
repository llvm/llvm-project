; RUN: llc -mtriple=s390x-ibm-zos -filetype=obj -o %t %s
; RUN: llvm-nm %t | FileCheck %s

define i32 @foo() {
entry:
  ret i32 0
}

; CHECK: t csect-default#C
; CHECK: d csect-default#S
