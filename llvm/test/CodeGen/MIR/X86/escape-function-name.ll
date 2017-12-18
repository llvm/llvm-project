; RUN: llc -mtriple=x86_64-unknown-unknown -stop-after branch-folder -o - %s 2>&1 | FileCheck %s

define void @"\01?f@@YAXXZ"() {
; CHECK: name: "\x01?f@@YAXXZ"
  ret void
}
