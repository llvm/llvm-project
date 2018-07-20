; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-p:32:32-n32-S64"

; CHECK-LABEL: @foo_ptr
; CHECK: and
define i32 @foo_ptr() {
entry:
  ; Even though the address of @foo is aligned, we cannot assume that the
  ; pointer has the same alignment. This is not true for e.g. ARM targets
  ; which store ARM/Thumb state in the LSB
  ret i32 and (i32 ptrtoint (void ()* @foo to i32), i32 -4)
}

define internal void @foo() align 16 {
entry:
  ret void
}
