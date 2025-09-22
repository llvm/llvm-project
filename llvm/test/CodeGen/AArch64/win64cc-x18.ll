;; Testing that nest uses x15 on all calling conventions (except Arm64EC)

; RUN: llc < %s -mtriple=aarch64-pc-windows-msvc | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-apple-darwin- | FileCheck %s

define dso_local i64 @other(ptr nest %p) #0 {
; CHECK-LABEL: other:
; CHECK:    ldr x0, [x15]
; CHECK:    ret
  %r = load i64, ptr %p
  ret i64 %r
}

define dso_local void @func() #0 {
; CHECK-LABEL: func:
; CHECK:    add x15, sp, #8
; CHECK:    bl {{_?other}}
; CHECK:    ret
entry:
  %p = alloca i64
  store i64 1, ptr %p
  call void @other(ptr nest %p)
  ret void
}

attributes #0 = { nounwind }
