; RUN: echo 'foo' > %t
; RUN: not --crash opt -S -extract-blocks -extract-blocks-file=%t %s 2>&1 | FileCheck %s

; CHECK: Invalid line
define void @bar() {
bb:
  ret void
}

