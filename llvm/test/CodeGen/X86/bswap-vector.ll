; RUN: llc < %s -mcpu=core | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare <2 x i64> @llvm.bswap.v2i64(<2 x i64>)

define <2 x i64> @foo(<2 x i64> %v) #0 {
entry:
  %r = call <2 x i64> @llvm.bswap.v2i64(<2 x i64> %v)
  ret <2 x i64> %r
}

; CHECK-LABEL: @foo
; CHECK: bswapq
; CHECK: bswapq
; CHECK: ret

attributes #0 = { nounwind uwtable }

