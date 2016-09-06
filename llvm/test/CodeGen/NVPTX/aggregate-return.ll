; RUN: llc < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s

declare <2 x float> @barv(<2 x float> %input)
declare [2 x float] @bara([2 x float] %input)
declare {float, float} @bars({float, float} %input)

define void @foov(<2 x float> %input, <2 x float>* %output) {
; CHECK-LABEL: @foov
  %call = tail call <2 x float> @barv(<2 x float> %input)
; CHECK: .param .align 8 .b8 retval0[8];
; CHECK: ld.param.v2.f32 {[[ELEMV1:%f[0-9]+]], [[ELEMV2:%f[0-9]+]]}, [retval0+0];
  store <2 x float> %call, <2 x float>* %output, align 8
; CHECK: st.v2.f32 [{{%rd[0-9]+}}], {[[ELEMV1]], [[ELEMV2]]}
  ret void
}

define void @fooa([2 x float] %input, [2 x float]* %output) {
; CHECK-LABEL: @fooa
  %call = tail call [2 x float] @bara([2 x float] %input)
; CHECK: .param .align 4 .b8 retval0[8];
; CHECK-DAG: ld.param.f32 [[ELEMA1:%f[0-9]+]], [retval0+0];
; CHECK-DAG: ld.param.f32 [[ELEMA2:%f[0-9]+]], [retval0+4];
  store [2 x float] %call, [2 x float]* %output, align 4
; CHECK: }
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}], [[ELEMA1]]
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}+4], [[ELEMA2]]
  ret void
; CHECK: ret
}

define void @foos({float, float} %input, {float, float}* %output) {
; CHECK-LABEL: @foos
  %call = tail call {float, float} @bars({float, float} %input)
; CHECK: .param .align 4 .b8 retval0[8];
; CHECK-DAG: ld.param.f32 [[ELEMS1:%f[0-9]+]], [retval0+0];
; CHECK-DAG: ld.param.f32 [[ELEMS2:%f[0-9]+]], [retval0+4];
  store {float, float} %call, {float, float}* %output, align 4
; CHECK: }
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}], [[ELEMS1]]
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}+4], [[ELEMS2]]
  ret void
; CHECK: ret
}
