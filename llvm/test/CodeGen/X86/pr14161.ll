; RUN: llc < %s -mtriple=x86_64-linux-pc -mcpu=corei7 | FileCheck %s

declare <4 x i32> @llvm.x86.sse41.pminud(<4 x i32>, <4 x i32>)

define <2 x i16> @good(<4 x i32>*, <4 x i8>*) {
entry:
  %2 = load <4 x i32>* %0, align 16
  %3 = call <4 x i32> @llvm.x86.sse41.pminud(<4 x i32> %2, <4 x i32> <i32 127, i32 127, i32 127, i32 127>)
  %4 = extractelement <4 x i32> %3, i32 0
  %5 = extractelement <4 x i32> %3, i32 1
  %6 = extractelement <4 x i32> %3, i32 2
  %7 = extractelement <4 x i32> %3, i32 3
  %8 = bitcast i32 %4 to <2 x i16>
  %9 = bitcast i32 %5 to <2 x i16>
  ret <2 x i16> %8
; CHECK: good
; CHECK: pminud
; CHECK-NEXT: pmovzxwq
; CHECK: ret
}

define <2 x i16> @bad(<4 x i32>*, <4 x i8>*) {
entry:
  %2 = load <4 x i32>* %0, align 16
  %3 = call <4 x i32> @llvm.x86.sse41.pminud(<4 x i32> %2, <4 x i32> <i32 127, i32 127, i32 127, i32 127>)
  %4 = extractelement <4 x i32> %3, i32 0
  %5 = extractelement <4 x i32> %3, i32 1
  %6 = extractelement <4 x i32> %3, i32 2
  %7 = extractelement <4 x i32> %3, i32 3
  %8 = bitcast i32 %4 to <2 x i16>
  %9 = bitcast i32 %5 to <2 x i16>
  ret <2 x i16> %9
; CHECK: bad
; CHECK: pminud
; CHECK: pextrd
; CHECK: pmovzxwq
; CHECK: ret
}
