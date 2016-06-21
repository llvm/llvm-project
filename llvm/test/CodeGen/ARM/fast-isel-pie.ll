; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=pic -mtriple=armv7-pc-linux-gnueabi | FileCheck %s

@var = global i32 42

define i32* @foo() {
; CHECK:      foo:
; CHECK:      ldr     r0, .L[[POOL:.*]]
; CHECK-NEXT: .L[[ADDR:.*]]:
; CHECK-NEXT: add     r0, pc, r0
; CHECK-NEXT: bx      lr

; CHECK:      .L[[POOL]]:
; CHECK-NEXT: .long   var-(.L[[ADDR]]+8)

  ret i32* @var
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"PIE Level", i32 2}
