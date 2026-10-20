; RUN: opt -S -print-inst-addrs %s | FileCheck %s

define void @foo() {
  ; CHECK: ret void ; 0x
  ret void
}
