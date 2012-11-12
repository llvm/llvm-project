; RUN: opt -scalarrepl -S < %s | FileCheck %s
; rdar://9786827

; SROA should be able to handle the mixed types and eliminate the allocas here.

; TODO: Currently it does this by falling back to integer "bags of bits".
; With enough cleverness, it should be possible to convert between <3 x i32>
; and <2 x i64> by using a combination of a bitcast and a shuffle.

; CHECK: {
; CHECK-NOT: alloca
; CHECK: }

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

define <2 x i64> @foo() nounwind {
entry:
  %retval = alloca <3 x i32>, align 16
  %z = alloca <4 x i32>, align 16
  %tmp = load <4 x i32>* %z
  %tmp1 = shufflevector <4 x i32> %tmp, <4 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
  store <3 x i32> %tmp1, <3 x i32>* %retval
  %0 = bitcast <3 x i32>* %retval to <2 x i64>*
  %1 = load <2 x i64>* %0, align 1
  ret <2 x i64> %1
}
