; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -O0 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -O0 | %ptxas-verify %}

; Expanding these stores produces a truncating store from i128 to i1. It
; must be expanded by the generic legalizer, not lowered as a predicate store.

define void @store_v17i113(ptr %p) {
; CHECK-LABEL: store_v17i113(
; CHECK: ld.param.b64 [[VEC_DST:%rd[0-9]+]], [store_v17i113_param_0];
; CHECK: st.b8 [[[VEC_DST]]+240], 0;
; CHECK: ret;
  store <17 x i113> zeroinitializer, ptr %p
  ret void
}

define void @store_i129(ptr %src, ptr %dst) {
; CHECK-LABEL: store_i129(
; CHECK-DAG: ld.param.b64 [[SRC:%rd[0-9]+]], [store_i129_param_0];
; CHECK-DAG: ld.param.b64 [[DST:%rd[0-9]+]], [store_i129_param_1];
; CHECK-DAG: ld.b8 [[TAIL:%rd[0-9]+]], [[[SRC]]+16];
; CHECK-DAG: ld.v2.b64 {[[LO:%rd[0-9]+]], [[HI:%rd[0-9]+]]}, [[[SRC]]];
; CHECK-DAG: st.v2.b64 [[[DST]]], {[[LO]], [[HI]]};
; CHECK-DAG: and.b64 [[BIT:%rd[0-9]+]], [[TAIL]], 1;
; CHECK-DAG: st.b8 [[[DST]]+16], [[BIT]];
; CHECK: ret;
  %v = load i129, ptr %src
  store i129 %v, ptr %dst
  ret void
}

define void @store_i1(ptr %p, i1 %v) {
; CHECK-LABEL: store_i1(
; CHECK-DAG: ld.param.b64 [[BOOL_DST:%rd[0-9]+]], [store_i1_param_0];
; CHECK-DAG: ld.param.b8 [[BOOL_IN:%rs[0-9]+]], [store_i1_param_1];
; CHECK-DAG: and.b16 [[BOOL_BIT:%rs[0-9]+]], [[BOOL_IN]], 1;
; CHECK: st.b8 [[[BOOL_DST]]], [[BOOL_BIT]];
; CHECK: ret;
  store i1 %v, ptr %p
  ret void
}
