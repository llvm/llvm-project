; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define <16 x float> @test_v16f32(<16 x float> %a) {
; CHECK-LABEL: test_v16f32(
; CHECK-DAG: ld.param.v4.b32     {[[V_12_15:(%r[0-9]+[, ]*){4}]]}, [test_v16f32_param_0+48];
; CHECK-DAG: ld.param.v4.b32     {[[V_8_11:(%r[0-9]+[, ]*){4}]]}, [test_v16f32_param_0+32];
; CHECK-DAG: ld.param.v4.b32     {[[V_4_7:(%r[0-9]+[, ]*){4}]]}, [test_v16f32_param_0+16];
; CHECK-DAG: ld.param.v4.b32     {[[V_0_3:(%r[0-9]+[, ]*){4}]]}, [test_v16f32_param_0];
; CHECK-DAG: st.param.v4.b32     [func_retval0],  {[[V_0_3]]}
; CHECK-DAG: st.param.v4.b32     [func_retval0+16], {[[V_4_7]]}
; CHECK-DAG: st.param.v4.b32     [func_retval0+32], {[[V_8_11]]}
; CHECK-DAG: st.param.v4.b32     [func_retval0+48], {[[V_12_15]]}
; CHECK: ret;
  ret <16 x float> %a
}

define <8 x float> @test_v8f32(<8 x float> %a) {
; CHECK-LABEL: test_v8f32(
; CHECK-DAG: ld.param.v4.b32     {[[V_4_7:(%r[0-9]+[, ]*){4}]]}, [test_v8f32_param_0+16];
; CHECK-DAG: ld.param.v4.b32     {[[V_0_3:(%r[0-9]+[, ]*){4}]]}, [test_v8f32_param_0];
; CHECK-DAG: st.param.v4.b32     [func_retval0],  {[[V_0_3]]}
; CHECK-DAG: st.param.v4.b32     [func_retval0+16], {[[V_4_7]]}
; CHECK: ret;
  ret <8 x float> %a
}

define <4 x float> @test_v4f32(<4 x float> %a) {
; CHECK-LABEL: test_v4f32(
; CHECK-DAG: ld.param.v4.b32     {[[V_0_3:(%r[0-9]+[, ]*){4}]]}, [test_v4f32_param_0];
; CHECK-DAG: st.param.v4.b32     [func_retval0],  {[[V_0_3]]}
; CHECK: ret;
  ret <4 x float> %a
}

define <2 x float> @test_v2f32(<2 x float> %a) {
; CHECK-LABEL: test_v2f32(
; CHECK-DAG: ld.param.v2.b32     {[[V_0_3:(%r[0-9]+[, ]*){2}]]}, [test_v2f32_param_0];
; CHECK-DAG: st.param.v2.b32     [func_retval0],  {[[V_0_3]]}
; CHECK: ret;
  ret <2 x float> %a
}

; Oddly shaped vectors should not load any extra elements.
define <3 x float> @test_v3f32(<3 x float> %a) {
; CHECK-LABEL: test_v3f32(
; CHECK-DAG: ld.param.b32        [[V_2:%r[0-9]+]], [test_v3f32_param_0+8];
; CHECK-DAG: ld.param.v2.b32     {[[V_0_1:(%r[0-9]+[, ]*){2}]]}, [test_v3f32_param_0];
; CHECK-DAG: st.param.v2.b32     [func_retval0], {[[V_0_1]]}
; CHECK-DAG: st.param.b32        [func_retval0+8], [[V_2]]
; CHECK: ret;
  ret <3 x float> %a
}

define <8 x i64> @test_v8i64(<8 x i64> %a) {
; CHECK-LABEL: test_v8i64(
; CHECK-DAG: ld.param.v2.b64     {[[V_6_7:(%rd[0-9]+[, ]*){2}]]}, [test_v8i64_param_0+48];
; CHECK-DAG: ld.param.v2.b64     {[[V_4_5:(%rd[0-9]+[, ]*){2}]]}, [test_v8i64_param_0+32];
; CHECK-DAG: ld.param.v2.b64     {[[V_2_3:(%rd[0-9]+[, ]*){2}]]}, [test_v8i64_param_0+16];
; CHECK-DAG: ld.param.v2.b64     {[[V_0_1:(%rd[0-9]+[, ]*){2}]]}, [test_v8i64_param_0];
; CHECK-DAG: st.param.v2.b64     [func_retval0],  {[[V_0_1]]}
; CHECK-DAG: st.param.v2.b64     [func_retval0+16], {[[V_2_3]]}
; CHECK-DAG: st.param.v2.b64     [func_retval0+32], {[[V_4_5]]}
; CHECK-DAG: st.param.v2.b64     [func_retval0+48], {[[V_6_7]]}
; CHECK: ret;
  ret <8 x i64> %a
}

define <16 x i16> @test_v16i16(<16 x i16> %a) {
; CHECK-LABEL: test_v16i16(
; CHECK-DAG: ld.param.v4.b32     {[[V_8_15:(%r[0-9]+[, ]*){4}]]}, [test_v16i16_param_0+16];
; CHECK-DAG: ld.param.v4.b32     {[[V_0_7:(%r[0-9]+[, ]*){4}]]}, [test_v16i16_param_0];
; CHECK-DAG: st.param.v4.b32     [func_retval0], {[[V_0_7]]}
; CHECK-DAG: st.param.v4.b32     [func_retval0+16], {[[V_8_15]]}
; CHECK: ret;
  ret <16 x i16> %a
}
