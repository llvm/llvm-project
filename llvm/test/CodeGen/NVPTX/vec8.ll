; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target triple = "nvptx-unknown-cuda"

; CHECK: .visible .func foo
define void @foo(<8 x i8> %a, ptr %b) {
; CHECK-DAG: ld.param.v2.u32 {[[E0:%r[0-9]+]], [[E1:%r[0-9]+]]}, [foo_param_0]
; CHECK-DAG: ld.param.u64   %[[B:rd[0-9+]]], [foo_param_1]
; CHECK:     add.s16        [[T:%rs[0-9+]]],
; CHECK:     st.u8          [%[[B]]], [[T]];
  %t0 = extractelement <8 x i8> %a, i32 1
  %t1 = extractelement <8 x i8> %a, i32 6
  %t  = add i8 %t0, %t1
  store i8 %t, ptr %b
  ret void
}

