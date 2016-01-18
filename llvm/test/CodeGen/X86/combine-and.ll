; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=+sse4.1 < %s | FileCheck %s
;
; Verify that the DAGCombiner is able to fold a vector AND into a blend
; if one of the operands to the AND is a vector of all constants, and each
; constant element is either zero or all-ones.


define <4 x i32> @test1(<4 x i32> %A) {
; CHECK-LABEL: test1:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3,4,5,6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 0, i32 0>
  ret <4 x i32> %1
}

define <4 x i32> @test2(<4 x i32> %A) {
; CHECK-LABEL: test2:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3],xmm1[4,5,6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 0, i32 0>
  ret <4 x i32> %1
}

define <4 x i32> @test3(<4 x i32> %A) {
; CHECK-LABEL: test3:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5],xmm1[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 0, i32 -1, i32 0>
  ret <4 x i32> %1
}

define <4 x i32> @test4(<4 x i32> %A) {
; CHECK-LABEL: test4:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3,4,5],xmm0[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 0, i32 0, i32 -1>
  ret <4 x i32> %1
}

define <4 x i32> @test5(<4 x i32> %A) {
; CHECK-LABEL: test5:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5],xmm1[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 0>
  ret <4 x i32> %1
}

define <4 x i32> @test6(<4 x i32> %A) {
; CHECK-LABEL: test6:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3],xmm1[4,5],xmm0[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 0, i32 -1>
  ret <4 x i32> %1
}

define <4 x i32> @test7(<4 x i32> %A) {
; CHECK-LABEL: test7:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1,2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 0, i32 -1, i32 -1>
  ret <4 x i32> %1
}

define <4 x i32> @test8(<4 x i32> %A) {
; CHECK-LABEL: test8:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3,4,5],xmm0[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 0, i32 -1>
  ret <4 x i32> %1
}

define <4 x i32> @test9(<4 x i32> %A) {
; CHECK-LABEL: test9:
; CHECK:       # BB#0:
; CHECK-NEXT:    movq {{.*#+}} xmm0 = xmm0[0],zero
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 -1, i32 0, i32 0>
  ret <4 x i32> %1
}

define <4 x i32> @test10(<4 x i32> %A) {
; CHECK-LABEL: test10:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5],xmm1[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 -1, i32 0>
  ret <4 x i32> %1
}

define <4 x i32> @test11(<4 x i32> %A) {
; CHECK-LABEL: test11:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3,4,5,6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %1
}

define <4 x i32> @test12(<4 x i32> %A) {
; CHECK-LABEL: test12:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3,4,5],xmm1[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 -1, i32 -1, i32 0>
  ret <4 x i32> %1
}

define <4 x i32> @test13(<4 x i32> %A) {
; CHECK-LABEL: test13:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1,2,3],xmm1[4,5],xmm0[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 -1, i32 0, i32 -1>
  ret <4 x i32> %1
}

define <4 x i32> @test14(<4 x i32> %A) {
; CHECK-LABEL: test14:
; CHECK:       # BB#0:
; CHECK-NEXT:    pxor %xmm1, %xmm1
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 -1>
  ret <4 x i32> %1
}

define <4 x i32> @test15(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: test15:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5,6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 -1>
  %2 = and <4 x i32> %B, <i32 0, i32 -1, i32 0, i32 0>
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @test16(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: test16:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm0[0,1],xmm1[2,3],xmm0[4,5],xmm1[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 -1, i32 0, i32 -1, i32 0>
  %2 = and <4 x i32> %B, <i32 0, i32 -1, i32 0, i32 -1>
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}

define <4 x i32> @test17(<4 x i32> %A, <4 x i32> %B) {
; CHECK-LABEL: test17:
; CHECK:       # BB#0:
; CHECK-NEXT:    pblendw {{.*#+}} xmm0 = xmm1[0,1],xmm0[2,3],xmm1[4,5],xmm0[6,7]
; CHECK-NEXT:    retq
  %1 = and <4 x i32> %A, <i32 0, i32 -1, i32 0, i32 -1>
  %2 = and <4 x i32> %B, <i32 -1, i32 0, i32 -1, i32 0>
  %3 = or <4 x i32> %1, %2
  ret <4 x i32> %3
}
