; RUN: llc  -mtriple=armv7-pc-linux-gnueabi -relocation-model=pic < %s | FileCheck %s

@foo = dso_local global i32 42

define dso_local ptr @get_foo() {
  ret ptr @foo
}

; Test that we only use one load. Even that is only needed because there
; doesn't seem to be pc relative relocations for movw movt.
; CHECK:      ldr     r0, .LCPI0_0
; CHECK-NEXT: .L{{.*}}:
; CHECK-NEXT: add     r0, pc, r0
; CHECK-NEXT: bx      lr

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"PIE Level", i32 2}
