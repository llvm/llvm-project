; RUN: llc < %s -mtriple=i386-pc-win32 -mcpu=corei7 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 | FileCheck %s --check-prefix=X64

; ModuleID = 'a.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32"
target triple = "i386-pc-win32"

%v4_varying_big_struct = type { [4 x <4 x i32>] }

declare <4 x i32> @"foo"(%v4_varying_big_struct, <4 x i32>) nounwind

define <4 x i32> @"bar"(%v4_varying_big_struct %s, <4 x i32> %__mask) nounwind {
allocas:
  %calltmp = call <4 x i32> @"foo"(%v4_varying_big_struct %s, <4 x i32> %__mask)
  ret <4 x i32> %calltmp
; CHECK: bar
; CHECK: andl
; CHECK: call
; CHECK: ret
}

declare <8 x float> @bar64(<8 x float> %i0, <8 x float> %i1,
                         <8 x float> %i2, <8 x float> %i3,
                         <8 x float> %i4, <8 x float> %i5,
                         <8 x float> %i6, <8 x float> %i7,
                         <8 x float> %i8, <8 x float> %i9)

define <8 x float> @foo64(ptr %p) {
  %1 = load <8 x float>, ptr %p
  %idx1 = getelementptr inbounds <8 x float>, ptr %p, i64 1
  %2 = load <8 x float>, ptr %idx1
  %idx2 = getelementptr inbounds <8 x float>, ptr %p, i64 2
  %3 = load <8 x float>, ptr %idx2
  %idx3 = getelementptr inbounds <8 x float>, ptr %p, i64 3
  %4 = load <8 x float>, ptr %idx3
  %idx4 = getelementptr inbounds <8 x float>, ptr %p, i64 4
  %5 = load <8 x float>, ptr %idx4
  %idx5 = getelementptr inbounds <8 x float>, ptr %p, i64 5
  %6 = load <8 x float>, ptr %idx5
  %idx6 = getelementptr inbounds <8 x float>, ptr %p, i64 6
  %7 = load <8 x float>, ptr %idx6
  %idx7 = getelementptr inbounds <8 x float>, ptr %p, i64 7
  %8 = load <8 x float>, ptr %idx7
  %idx8 = getelementptr inbounds <8 x float>, ptr %p, i64 8
  %9 = load <8 x float>, ptr %idx8
  %idx9 = getelementptr inbounds <8 x float>, ptr %p, i64 9
  %10 = load <8 x float>, ptr %idx9
  %r = tail call <8 x float> @bar64(<8 x float> %1, <8 x float> %2,
                                    <8 x float> %3, <8 x float> %4,
                                    <8 x float> %5, <8 x float> %6,
                                    <8 x float> %7, <8 x float> %8,
                                    <8 x float> %9, <8 x float> %10)
  ret <8 x float> %r
; X64: foo
; X64: and
; X64: call
; X64: ret
}
