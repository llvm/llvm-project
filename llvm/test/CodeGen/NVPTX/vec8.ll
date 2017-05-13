; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target triple = "nvptx-unknown-cuda"

; CHECK: .visible .func foo
define void @foo(<8 x i8> %a, i8* %b) {
; CHECK-DAG: ld.param.v4.u8 {[[E0:%rs[0-9]+]], [[E1:%rs[0-9]+]], [[E2:%rs[0-9]+]], [[E3:%rs[0-9]+]]}, [foo_param_0]
; CHECK-DAG: ld.param.v4.u8 {[[E4:%rs[0-9]+]], [[E5:%rs[0-9]+]], [[E6:%rs[0-9]+]], [[E7:%rs[0-9]+]]}, [foo_param_0+4]
; CHECK-DAG: ld.param.u32   %[[B:r[0-9+]]], [foo_param_1]
; CHECK-DAG: add.s16        [[T:%rs[0-9+]]], [[E1]], [[E6]];
; CHECK:     st.u8          [%[[B]]], [[T]];
  %t0 = extractelement <8 x i8> %a, i32 1
  %t1 = extractelement <8 x i8> %a, i32 6
  %t  = add i8 %t0, %t1
  store i8 %t, i8* %b
  ret void
}

