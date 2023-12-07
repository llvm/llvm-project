; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-NOT: .align 2
define ptx_device void @foo() align 2 {
; CHECK-LABEL: .func foo
  ret void
}
