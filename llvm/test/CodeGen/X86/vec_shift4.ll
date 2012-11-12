; RUN: llc < %s -march=x86 -mattr=+sse41 | FileCheck %s

define <2 x i64> @shl1(<4 x i32> %r, <4 x i32> %a) nounwind readnone ssp {
entry:
; CHECK-NOT: shll
; CHECK: pslld
; CHECK: paddd
; CHECK: cvttps2dq
; CHECK: pmulld

  %shl = shl <4 x i32> %r, %a                     ; <<4 x i32>> [#uses=1]
  %tmp2 = bitcast <4 x i32> %shl to <2 x i64>     ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %tmp2
}

define <2 x i64> @shl2(<16 x i8> %r, <16 x i8> %a) nounwind readnone ssp {
entry:
; CHECK-NOT: shlb
; CHECK: pblendvb
; CHECK: pblendvb
; CHECK: pblendvb
  %shl = shl <16 x i8> %r, %a                     ; <<16 x i8>> [#uses=1]
  %tmp2 = bitcast <16 x i8> %shl to <2 x i64>     ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %tmp2
}
