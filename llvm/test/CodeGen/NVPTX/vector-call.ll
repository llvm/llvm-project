; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target triple = "nvptx-unknown-cuda"

declare void @bar(<4 x i32>)

; CHECK-LABEL: .func foo(
; CHECK-DAG: ld.param.v4.u32 {[[E0:%r[0-9]+]], [[E1:%r[0-9]+]], [[E2:%r[0-9]+]], [[E3:%r[0-9]+]]}, [foo_param_0];
; CHECK: .param .align 16 .b8 param0[16];
; CHECK-DAG: st.param.v4.b32  [param0],  {[[E0]], [[E1]], [[E2]], [[E3]]};
; CHECK:     call.uni
; CHECK:     ret;
define void @foo(<4 x i32> %a) {
  tail call void @bar(<4 x i32> %a)
  ret void
}

; CHECK-LABEL: .func foo3(
; CHECK-DAG: ld.param.v2.u32 {[[E0:%r[0-9]+]], [[E1:%r[0-9]+]]}, [foo3_param_0];
; CHECK-DAG: ld.param.u32 [[E2:%r[0-9]+]], [foo3_param_0+8];
; CHECK: .param .align 16 .b8 param0[16];
; CHECK-DAG: st.param.v2.b32  [param0],  {[[E0]], [[E1]]};
; CHECK-DAG: st.param.b32     [param0+8],  [[E2]];
; CHECK:     call.uni
; CHECK:     ret;
declare void @bar3(<3 x i32>)
define void @foo3(<3 x i32> %a) {
  tail call void @bar3(<3 x i32> %a)
  ret void
}
