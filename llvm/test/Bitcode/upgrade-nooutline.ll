; RUN: llvm-as < %s | llvm-dis - | FileCheck %s

; CHECK: define void @f() [[ATTR:#[0-9]+]]
; CHECK-NOT: "nooutline"
; CHECK: attributes [[ATTR]] = {
; CHECK-NOT: "nooutline"
; CHECK-SAME: nooutline
; CHECK-NOT: "nooutline"

define void @f() "nooutline" {
  ret void
}
