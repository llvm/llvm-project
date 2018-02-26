; Inlining should not clone label annotations.
; Currently we block all duplication for simplicity.

; RUN: opt < %s -S -inline | FileCheck %s

@the_global = global i32 0

declare void @llvm.codeview.annotation(metadata)

define void @inlinee() {
entry:
  store i32 42, i32* @the_global
  call void @llvm.codeview.annotation(metadata !0)
  ret void
}

define void @caller() {
entry:
  call void @inlinee()
  ret void
}

!0 = !{!"annotation"}

; CHECK-LABEL: define void @inlinee()
; CHECK: store i32 42, i32* @the_global
; CHECK: call void @llvm.codeview.annotation(metadata !0)
; CHECK: ret void

; CHECK-LABEL: define void @caller()
;       MSVC can inline this. If we ever do, check for the store but make sure
;       there is no annotation.
; CHECK: call void @inlinee()
; CHECK-NOT: call void @llvm.codeview.annotation
; CHECK: ret void
