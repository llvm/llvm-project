; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK: // .weak foo
; CHECK: .weak .func foo
define weak void @foo() {
  ret void
}

; CHECK: // .weak baz
; CHECK: .weak .func baz
define weak_odr void @baz() {
  ret void
}

; CHECK: .visible .func bar
define void @bar() {
  ret void
}
