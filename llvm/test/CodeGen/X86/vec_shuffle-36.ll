; RUN: llc < %s -march=x86-64 -mcpu=penryn -mattr=sse41 | FileCheck %s

define <8 x i16> @shuf6(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
; CHECK: pshufb
; CHECK-NOT: pshufb
; CHECK: ret
entry:
  %tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 3, i32 2, i32 0, i32 2, i32 1, i32 5, i32 6 , i32 undef >
  ret <8 x i16> %tmp9
}

define <8 x i16> @shuf7(<8 x i16> %t0) {
; CHECK: pshufd
  %tmp10 = shufflevector <8 x i16> %t0, <8 x i16> undef, <8 x i32> < i32 undef, i32 2, i32 2, i32 2, i32 2, i32 2, i32 undef, i32 undef >
  ret <8 x i16> %tmp10
}
