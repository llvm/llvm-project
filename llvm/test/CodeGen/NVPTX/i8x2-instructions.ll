; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx80 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN: | FileCheck  %s
; RUN: %if ptxas %{                                                           \
; RUN:   llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_90 -asm-verbose=false \
; RUN:          -O0 -disable-post-ra -frame-pointer=all -verify-machineinstrs \
; RUN:   | %ptxas-verify -arch=sm_90                                          \
; RUN: %}

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; CHECK-LABEL: test_trunc_2xi8(
; CHECK:      ld.param.u32 [[R1:%r[0-9]+]], [test_trunc_2xi8_param_0];
; CHECK:      mov.b32 {[[RS1:%rs[0-9]+]], [[RS2:%rs[0-9]+]]}, [[R1]];
; CHECK:      shl.b16 	[[RS3:%rs[0-9]+]], [[RS2]], 8;
; CHECK:      and.b16  [[RS4:%rs[0-9]+]], [[RS1]], 255;
; CHECK:      or.b16   [[RS5:%rs[0-9]+]], [[RS4]], [[RS3]]
; CHECK:      cvt.u32.u16  [[R2:%r[0-9]]], [[RS5]]
; CHECK:      st.param.b32  [func_retval0], [[R2]];
define i16 @test_trunc_2xi8(<2 x i16> %a) #0 {
  %trunc = trunc <2 x i16> %a to <2 x i8>
  %res = bitcast <2 x i8> %trunc to i16
  ret i16 %res
}

; CHECK-LABEL: test_zext_2xi8(
; CHECK:      ld.param.u16  [[RS1:%rs[0-9]+]], [test_zext_2xi8_param_0];
; CHECK:      shr.u16 	[[RS2:%rs[0-9]+]], [[RS1]], 8;
; CHECK:      mov.b32  [[R1:%r[0-9]+]], {[[RS1]], [[RS2]]}
; CHECK:      and.b32  [[R2:%r[0-9]+]], [[R1]], 16711935;
; CHECK:      st.param.b32  [func_retval0], [[R2]];
define <2 x i16> @test_zext_2xi8(i16 %a) #0 {
  %vec = bitcast i16 %a to <2 x i8>
  %ext = zext <2 x i8> %vec to <2 x i16>
  ret <2 x i16> %ext
}
