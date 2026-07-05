; RUN: llc < %s -mtriple=thumb | FileCheck %s

; CHECK: .code	16

define void @f() {
  ret void
}
