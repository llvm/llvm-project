; RUN: opt %s -inline -S | FileCheck %s

define internal void @inner() "probe-stack"="__probestackinner" {
  ret void
}

define void @outerNoAttribute() {
  call void @inner()
  ret void
}

define void @outerConflictingAttribute() "probe-stack"="__probestackouter" {
  call void @inner()
  ret void
}

; CHECK: define void @outerNoAttribute() #0
; CHECK: define void @outerConflictingAttribute() #1
; CHECK: attributes #0 = { "probe-stack"="__probestackinner" }
; CHECK: attributes #1 = { "probe-stack"="__probestackouter" }
