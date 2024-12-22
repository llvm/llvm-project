; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx80 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck  %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN:   | %ptxas-verify -arch=sm_90                                          \
; RUN: %}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: test_bitcast_2xi8_i16(
; CHECK: ld.param.u32 	%r1, [test_bitcast_2xi8_i16_param_0];
; CHECK: mov.b32 	{%rs1, %rs2}, %r1;
; CHECK: shl.b16 	%rs3, %rs2, 8;
; CHECK: and.b16  	%rs4, %rs1, 255;
; CHECK: or.b16  	%rs5, %rs4, %rs3;
; CHECK: cvt.u32.u16 	%r2, %rs5;
; CHECK: st.param.b32 	[func_retval0], %r2;
define i16 @test_bitcast_2xi8_i16(<2 x i8> %a) {
  %res = bitcast <2 x i8> %a to i16
  ret i16 %res
}

; CHECK-LABEL: test_bitcast_i16_2xi8(
; CHECK: ld.param.u16 	%rs1, [test_bitcast_i16_2xi8_param_0];
; CHECK: shr.u16 	%rs2, %rs1, 8;
; CHECK: mov.b32 	%r1, {%rs1, %rs2};
; CHECK: st.param.b32 	[func_retval0], %r1;
define <2 x i8> @test_bitcast_i16_2xi8(i16 %a) {
  %res = bitcast i16 %a to <2 x i8>
  ret <2 x i8> %res
}
