; RUN: llc -mtriple=x86_64-apple-darwin -mattr=+sse2 < %s | FileCheck --check-prefix=SSE2-CODEGEN %s
; RUN: opt -mtriple=x86_64-apple-darwin -mattr=+sse2 -passes="print<cost-model>" 2>&1 -disable-output < %s | FileCheck --check-prefix=SSE2 %s

define <4 x i32> @shl(<4 x i32> %vector, i32 %scalar) {
entry:
  ; SSE2: 'shl'
  ; SSE2: cost of 2 {{.*}} shl
  ; SSE2-CODEGEN: movd  %edi, %xmm1
  ; SSE2-CODEGEN: pslld %xmm1, %xmm0
  %insert = insertelement <4 x i32> undef, i32 %scalar, i32 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  %ret = shl <4 x i32> %vector , %splat
  ret <4 x i32> %ret
}

define <4 x i32> @ashr(<4 x i32> %vector, i32 %scalar) {
entry:
  ; SSE2: 'ashr'
  ; SSE2: cost of 2 {{.*}} ashr
  ; SSE2-CODEGEN: movd  %edi, %xmm1
  ; SSE2-CODEGEN: psrad %xmm1, %xmm0
  %insert = insertelement <4 x i32> undef, i32 %scalar, i32 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  %ret = ashr <4 x i32> %vector , %splat
  ret <4 x i32> %ret
}

define <4 x i32> @lshr(<4 x i32> %vector, i32 %scalar) {
entry:
  ; SSE2: 'lshr'
  ; SSE2: cost of 2 {{.*}} lshr
  ; SSE2-CODEGEN: movd  %edi, %xmm1
  ; SSE2-CODEGEN: psrld %xmm1, %xmm0
  %insert = insertelement <4 x i32> undef, i32 %scalar, i32 0
  %splat = shufflevector <4 x i32> %insert, <4 x i32> undef, <4 x i32> zeroinitializer
  %ret = lshr <4 x i32> %vector , %splat
  ret <4 x i32> %ret
}

