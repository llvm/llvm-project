; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

declare <2 x float> @barv(<2 x float> %input)
declare <3 x float> @barv3(<3 x float> %input)
declare [2 x float] @bara([2 x float] %input)
declare {float, float} @bars({float, float} %input)

define void @test_v2f32(<2 x float> %input, ptr %output) {
; CHECK-LABEL: @test_v2f32
  %call = tail call <2 x float> @barv(<2 x float> %input)
; CHECK: .param .align 8 .b8 retval0[8];
; CHECK: ld.param.v2.f32 {[[E0:%f[0-9]+]], [[E1:%f[0-9]+]]}, [retval0];
  store <2 x float> %call, ptr %output, align 8
; CHECK: st.v2.f32 [{{%rd[0-9]+}}], {[[E0]], [[E1]]}
  ret void
}

define void @test_v3f32(<3 x float> %input, ptr %output) {
; CHECK-LABEL: @test_v3f32
;
  %call = tail call <3 x float> @barv3(<3 x float> %input)
; CHECK: .param .align 16 .b8 retval0[16];
; CHECK-DAG: ld.param.v2.f32 {[[E0:%f[0-9]+]], [[E1:%f[0-9]+]]}, [retval0];
; CHECK-DAG: ld.param.f32 [[E2:%f[0-9]+]], [retval0+8];
; Make sure we don't load more values than than we need to.
; CHECK-NOT: ld.param.f32 [[E3:%f[0-9]+]], [retval0+12];
  store <3 x float> %call, ptr %output, align 8
; CHECK-DAG: st.f32 [{{%rd[0-9]}}+8],
; -- This is suboptimal. We should do st.v2.f32 instead
;    of combining 2xf32 info i64.
; CHECK-DAG: st.u64 [{{%rd[0-9]}}],
; CHECK: ret;
  ret void
}

define void @test_a2f32([2 x float] %input, ptr %output) {
; CHECK-LABEL: @test_a2f32
  %call = tail call [2 x float] @bara([2 x float] %input)
; CHECK: .param .align 4 .b8 retval0[8];
; CHECK-DAG: ld.param.f32 [[ELEMA1:%f[0-9]+]], [retval0];
; CHECK-DAG: ld.param.f32 [[ELEMA2:%f[0-9]+]], [retval0+4];
  store [2 x float] %call, ptr %output, align 4
; CHECK: }
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}], [[ELEMA1]]
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}+4], [[ELEMA2]]
  ret void
; CHECK: ret
}

define void @test_s2f32({float, float} %input, ptr %output) {
; CHECK-LABEL: @test_s2f32
  %call = tail call {float, float} @bars({float, float} %input)
; CHECK: .param .align 4 .b8 retval0[8];
; CHECK-DAG: ld.param.f32 [[ELEMS1:%f[0-9]+]], [retval0];
; CHECK-DAG: ld.param.f32 [[ELEMS2:%f[0-9]+]], [retval0+4];
  store {float, float} %call, ptr %output, align 4
; CHECK: }
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}], [[ELEMS1]]
; CHECK-DAG: st.f32 [{{%rd[0-9]+}}+4], [[ELEMS2]]
  ret void
; CHECK: ret
}
