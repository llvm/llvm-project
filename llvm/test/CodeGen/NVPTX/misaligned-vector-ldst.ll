; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: t1
define <4 x float> @t1(ptr %p1) {
; CHECK-NOT: ld.v4
; CHECK-NOT: ld.v2
; CHECK-NOT: ld.f32
; CHECK: ld.u8
  %r = load <4 x float>, ptr %p1, align 1
  ret <4 x float> %r
}

; CHECK-LABEL: t2
define <4 x float> @t2(ptr %p1) {
; CHECK-NOT: ld.v4
; CHECK-NOT: ld.v2
; CHECK: ld.f32
  %r = load <4 x float>, ptr %p1, align 4
  ret <4 x float> %r
}

; CHECK-LABEL: t3
define <4 x float> @t3(ptr %p1) {
; CHECK-NOT: ld.v4
; CHECK: ld.v2
  %r = load <4 x float>, ptr %p1, align 8
  ret <4 x float> %r
}

; CHECK-LABEL: t4
define <4 x float> @t4(ptr %p1) {
; CHECK: ld.v4
  %r = load <4 x float>, ptr %p1, align 16
  ret <4 x float> %r
}

; CHECK-LABEL: .visible .func test_v1halfp0a1(
; CHECK-DAG: ld.param.u64 %[[FROM:rd?[0-9]+]], [test_v1halfp0a1_param_0];
; CHECK-DAG: ld.param.u64 %[[TO:rd?[0-9]+]], [test_v1halfp0a1_param_1];
; CHECK-DAG: ld.u8        [[B0:%r[sd]?[0-9]+]], [%[[FROM]]]
; CHECK-DAG: st.u8        [%[[TO]]], [[B0]]
; CHECK-DAG: ld.u8        [[B1:%r[sd]?[0-9]+]], [%[[FROM]]+1]
; CHECK-DAG: st.u8        [%[[TO]]+1], [[B1]]
; CHECK: ret
define void @test_v1halfp0a1(ptr noalias readonly %from, ptr %to) {
  %1 = load <1 x half>, ptr %from , align 1
  store <1 x half> %1, ptr %to , align 1
  ret void
}

; CHECK-LABEL: .visible .func test_v2halfp0a1(
; CHECK-DAG: ld.param.u64 %[[FROM:rd?[0-9]+]], [test_v2halfp0a1_param_0];
; CHECK-DAG: ld.param.u64 %[[TO:rd?[0-9]+]], [test_v2halfp0a1_param_1];
; CHECK-DAG: ld.u8        [[B0:%r[sd]?[0-9]+]], [%[[FROM]]]
; CHECK-DAG: st.u8        [%[[TO]]],
; CHECK-DAG: ld.u8        [[B1:%r[sd]?[0-9]+]], [%[[FROM]]+1]
; CHECK-DAG: st.u8        [%[[TO]]+1],
; CHECK-DAG: ld.u8        [[B2:%r[sd]?[0-9]+]], [%[[FROM]]+2]
; CHECK-DAG: st.u8        [%[[TO]]+2],
; CHECK-DAG: ld.u8        [[B3:%r[sd]?[0-9]+]], [%[[FROM]]+3]
; CHECK-DAG: st.u8        [%[[TO]]+3],
; CHECK: ret
define void @test_v2halfp0a1(ptr noalias readonly %from, ptr %to) {
  %1 = load <2 x half>, ptr %from , align 1
  store <2 x half> %1, ptr %to , align 1
  ret void
}

; CHECK-LABEL: .visible .func test_v4halfp0a1(
; CHECK-DAG: ld.param.u64 %[[FROM:rd?[0-9]+]], [test_v4halfp0a1_param_0];
; CHECK-DAG: ld.param.u64 %[[TO:rd?[0-9]+]], [test_v4halfp0a1_param_1];
; CHECK-DAG: ld.u8        [[B0:%r[sd]?[0-9]+]], [%[[FROM]]]
; CHECK-DAG: st.u8        [%[[TO]]], [[B0]]
; CHECK-DAG: ld.u8        [[B1:%r[sd]?[0-9]+]], [%[[FROM]]+1]
; CHECK-DAG: st.u8        [%[[TO]]+1], [[B1]]
; CHECK-DAG: ld.u8        [[B2:%r[sd]?[0-9]+]], [%[[FROM]]+2]
; CHECK-DAG: st.u8        [%[[TO]]+2], [[B2]]
; CHECK-DAG: ld.u8        [[B3:%r[sd]?[0-9]+]], [%[[FROM]]+3]
; CHECK-DAG: st.u8        [%[[TO]]+3], [[B3]]
; CHECK-DAG: ld.u8        [[B4:%r[sd]?[0-9]+]], [%[[FROM]]+4]
; CHECK-DAG: st.u8        [%[[TO]]+4], [[B4]]
; CHECK-DAG: ld.u8        [[B5:%r[sd]?[0-9]+]], [%[[FROM]]+5]
; CHECK-DAG: st.u8        [%[[TO]]+5], [[B5]]
; CHECK-DAG: ld.u8        [[B6:%r[sd]?[0-9]+]], [%[[FROM]]+6]
; CHECK-DAG: st.u8        [%[[TO]]+6], [[B6]]
; CHECK-DAG: ld.u8        [[B7:%r[sd]?[0-9]+]], [%[[FROM]]+7]
; CHECK-DAG: st.u8        [%[[TO]]+7], [[B7]]
; CHECK: ret
define void @test_v4halfp0a1(ptr noalias readonly %from, ptr %to) {
  %1 = load <4 x half>, ptr %from , align 1
  store <4 x half> %1, ptr %to , align 1
  ret void
}


; CHECK-LABEL: s1
define void @s1(ptr %p1, <4 x float> %v) {
; CHECK-NOT: st.v4
; CHECK-NOT: st.v2
; CHECK-NOT: st.f32
; CHECK: st.u8
  store <4 x float> %v, ptr %p1, align 1
  ret void
}

; CHECK-LABEL: s2
define void @s2(ptr %p1, <4 x float> %v) {
; CHECK-NOT: st.v4
; CHECK-NOT: st.v2
; CHECK: st.f32
  store <4 x float> %v, ptr %p1, align 4
  ret void
}

; CHECK-LABEL: s3
define void @s3(ptr %p1, <4 x float> %v) {
; CHECK-NOT: st.v4
  store <4 x float> %v, ptr %p1, align 8
  ret void
}

; CHECK-LABEL: s4
define void @s4(ptr %p1, <4 x float> %v) {
; CHECK: st.v4
  store <4 x float> %v, ptr %p1, align 16
  ret void
}

