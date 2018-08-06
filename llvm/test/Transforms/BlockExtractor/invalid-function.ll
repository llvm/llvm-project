; RUN: echo 'foo bb' > %t
; RUN: not opt -S -extract-blocks -extract-blocks-file=%t %s 2>&1 | FileCheck %s

; CHECK: Invalid function
define void @bar() {
bb:
  ret void
}

