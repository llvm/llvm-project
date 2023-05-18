; RUN: llc -mattr=+sse4.1 -mcpu=penryn < %s | FileCheck %s -check-prefix=CHECK-W-SSE4
; RUN: llc -mattr=-sse4.1 -mcpu=penryn < %s | FileCheck %s -check-prefix=CHECK-WO-SSE4
; Test case for r146671
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7"

define <16 x i8> @shift(<16 x i8> %a, <16 x i8> %b) nounwind {
  ; Make sure operands to pblend are in the right order.
  ; CHECK-W-SSE4: psllw $4, [[REG1:%xmm.]]
  ; CHECK-W-SSE4: pblendvb %xmm0, [[REG1]],{{ %xmm.}}
  ; CHECK-W-SSE4: psllw $2

  ; Make sure we're masking and pcmp'ing the VSELECT conditon vector.
  ; CHECK-WO-SSE4: psllw $5, [[REG1:%xmm.]]
  ; CHECK-WO-SSE4: pxor [[REG2:%xmm.]], [[REG2:%xmm.]]
  ; CHECK-WO-SSE4: pcmpgtb {{%xmm., }}[[REG2]]
  %1 = shl <16 x i8> %a, %b
  ret <16 x i8> %1
}
