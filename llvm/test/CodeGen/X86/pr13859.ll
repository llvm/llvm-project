; RUN: llc < %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.7.0"

define void @_Z17FilterYUVRows_MMXi(i32 %af) nounwind ssp {
entry:
  %aMyAlloca = alloca i32, align 32
  %dest = alloca <1 x i64>, align 32

  %a32 = load i32, ptr %aMyAlloca, align 4
  %aconv = trunc i32 %a32 to i16
  %a36 = insertelement <4 x i16> undef, i16 %aconv, i32 0
  %a37 = insertelement <4 x i16> %a36, i16 %aconv, i32 1
  %a38 = insertelement <4 x i16> %a37, i16 %aconv, i32 2
  %a39 = insertelement <4 x i16> %a38, i16 %aconv, i32 3
  %a40 = bitcast <4 x i16> %a39 to x86_mmx
  %a41 = bitcast x86_mmx %a40 to <1 x i64>

  %a47 = trunc i32 %a32 to i1
  br i1 %a47, label %a48, label %a49

a48:
  unreachable

a49:
  store <1 x i64> %a41, ptr %dest, align 8 ; !!!
  ret void
}
