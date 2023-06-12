; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

declare <4 x float> @bar()

; CHECK-LABEL: .func foo(
define void @foo(ptr %ptr) {
; CHECK:     ld.param.u64 %[[PTR:rd[0-9]+]], [foo_param_0];
; CHECK:     ld.param.v4.f32 {[[E0:%f[0-9]+]], [[E1:%f[0-9]+]], [[E2:%f[0-9]+]], [[E3:%f[0-9]+]]}, [retval0+0];
; CHECK:     st.v4.f32    [%[[PTR]]], {[[E0]], [[E1]], [[E2]], [[E3]]}
  %val = tail call <4 x float> @bar()
  store <4 x float> %val, ptr %ptr
  ret void
}
