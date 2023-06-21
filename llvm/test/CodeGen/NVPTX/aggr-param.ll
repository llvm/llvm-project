; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Make sure aggregate param types get emitted properly.

%struct.float4 = type { float, float, float, float }

; CHECK: .visible .func bar
; CHECK:   .param .align 4 .b8 bar_param_0[16]
define void @bar(%struct.float4 %f) {
entry:
  ret void
}

; CHECK: .visible .func foo
; CHECK:   .param .align 4 .b8 foo_param_0[20]
define void @foo([5 x i32] %f) {
entry:
  ret void
}

