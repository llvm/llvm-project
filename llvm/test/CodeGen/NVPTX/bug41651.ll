; RUN: llc -filetype=asm -o - %s | FileCheck %s
; RUN: %if ptxas %{ llc -filetype=asm -o - %s | %ptxas-verify %}

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%func = type { ptr }

; CHECK: foo
; CHECK: call
; CHECK: ret
define void @foo() {
  %call = call %func undef(i32 0, i32 1)
  ret void
}
