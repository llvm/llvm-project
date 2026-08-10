; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK: .visible .func  (.param .align 16 .b8 func_retval0[16]) foo0(
; CHECK:          .param .align 4 .b8 foo0_param_0[8]
define <4 x float> @foo0({float, float} %arg0) {
  ret <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>
}

; CHECK: .visible .func  (.param .align 8 .b8 func_retval0[8]) foo1(
; CHECK:          .param .align 8 .b8 foo1_param_0[16]
define <2 x float> @foo1({float, float, i64} %arg0) {
  ret <2 x float> <float 1.0, float 1.0>
}
