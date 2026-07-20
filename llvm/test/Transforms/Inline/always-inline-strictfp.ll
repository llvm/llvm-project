; RUN: opt -passes=always-inline,verify %s -S | FileCheck %s

declare void @foo() strictfp

define internal void @callee() alwaysinline strictfp {
entry:
  call void @foo()
  ret void
}

; CHECK: define void @caller() [[ATTR:#[0-9]+]]
; CHECK: call void @foo()
define void @caller() {
entry:
  call void @callee()
  ret void
}

; CHECK: attributes [[ATTR]] = {{{.*}}strictfp{{.*}}}
