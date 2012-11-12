; RUN: llc < %s -mtriple=x86_64-apple-darwin  -mcpu=corei7 | FileCheck %s
; ModuleID = '<stdin>'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin11.2.0"

; CHECK: @foo8
; CHECK: psll
; CHECK: psraw
; CHECK: pblendvb
; CHECK: ret
define void @foo8(float* nocapture %RET) nounwind {
allocas:
  %resultvec.i = select <8 x i1> <i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true>, <8 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>, <8 x i8> <i8 100, i8 100, i8 100, i8 100, i8 100, i8 100, i8 100, i8 100>
  %uint2float = uitofp <8 x i8> %resultvec.i to <8 x float>
  %ptr = bitcast float * %RET to <8 x float> *
  store <8 x float> %uint2float, <8 x float>* %ptr, align 4
  ret void
}


