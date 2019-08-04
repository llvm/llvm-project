; RUN: llc -filetype=asm -o - %s | FileCheck %s
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%func = type { i32 (i32, i32)** }

; CHECK: foo
; CHECK: call
; CHECK: ret
define void @foo() {
  %call = call %func undef(i32 0, i32 1)
  ret void
}
