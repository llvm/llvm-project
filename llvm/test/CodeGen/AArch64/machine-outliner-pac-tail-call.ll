; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-machine-outliner -verify-machineinstrs < %s | FileCheck %s

@g = common dso_local global i32 0, align 4
declare void @callee()

define dso_local void @foo(ptr %p) #0 {
; CHECK-LABEL: foo:
; CHECK: bl OUTLINED_FUNCTION_0
entry:
  store volatile i32 1, ptr %p, align 4
  store volatile i32 2, ptr %p, align 4
  call void @callee()
  store volatile i32 3, ptr @g, align 4
  ret void
}

define dso_local void @bar(ptr %p) #0 {
; CHECK-LABEL: bar:
; CHECK: bl OUTLINED_FUNCTION_0
entry:
  store volatile i32 1, ptr %p, align 4
  store volatile i32 2, ptr %p, align 4
  call void @callee()
  store volatile i32 4, ptr @g, align 4
  ret void
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK: hint #25
; CHECK: hint #29
; CHECK: b callee

attributes #0 = { "sign-return-address"="all" }
