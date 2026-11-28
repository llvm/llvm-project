; RUN: llc -mtriple=s390x-ibm-zos -filetype=obj -o %t %s
; RUN: llvm-nm %t | FileCheck %s

define i32 @foo() {
entry:
  ret i32 0
}

; CHECK:     {{[0-9a-f]+}} t {{ *$}}
; CHECK:     {{[0-9a-f]+}} d {{ *$}}
