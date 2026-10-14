; RUN: llc -mtriple=aarch64-unknown-linux-gnu -enable-machine-outliner -verify-machineinstrs < %s | FileCheck %s

@g = common dso_local global i32 0, align 4
declare void @callee()

define dso_local void @foo(i32 %sz) #0 {
; CHECK-LABEL: foo:
; CHECK: bl OUTLINED_FUNCTION_0
entry:
  %vla = alloca i32, i32 %sz, align 4
  store volatile i32 2, ptr %vla, align 4
  call void @callee()
  store volatile i32 3, ptr @g, align 4
  ret void
}

define dso_local void @bar(i32 %sz) #0 {
; CHECK-LABEL: bar:
; CHECK: bl OUTLINED_FUNCTION_0
entry:
  %vla = alloca i32, i32 %sz, align 4
  store volatile i32 2, ptr %vla, align 4
  call void @callee()
  store volatile i32 4, ptr @g, align 4
  ret void
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK-NOT: hint #25
; CHECK-NOT: paciasp
; CHECK: b callee

attributes #0 = { "sign-return-address"="all" }
