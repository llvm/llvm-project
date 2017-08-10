; RUN: llc -verify-machineinstrs < %s -mcpu=a2q | FileCheck %s
target triple = "powerpc64-bgq-linux"

@R = global <4 x i1> <i1 0, i1 0, i1 0, i1 0>, align 16

define <4 x float> @test1(<4 x float> %a, <4 x float> %b, <4 x i1> %c) nounwind readnone {
entry:
  %r = select <4 x i1> %c, <4 x float> %a, <4 x float> %b
  ret <4 x float> %r

; CHECK-LABEL: @test1
; CHECK: qvfsel 1, 3, 1, 2
; CHECK: blr
}

define <4 x float> @test2(<4 x float> %a, <4 x float> %b, i1 %c1, i1 %c2, i1 %c3, i1 %c4) nounwind readnone {
entry:
  %v = insertelement <4 x i1> undef, i1 %c1, i32 0
  %v2 = insertelement <4 x i1> %v, i1 %c2, i32 1
  %v3 = insertelement <4 x i1> %v2, i1 %c3, i32 2
  %v4 = insertelement <4 x i1> %v3, i1 %c4, i32 3
  %r = select <4 x i1> %v4, <4 x float> %a, <4 x float> %b
  ret <4 x float> %r

; CHECK-LABEL: @test2
; CHECK: stw
; CHECK-DAG: qvlfiwzx [[REG1:[0-9]+]],
; CHECK-DAG: qvlfdx [[REG2:[0-9]+]],
; CHECK-DAG: qvfcfidu [[REG3:[0-9]+]], [[REG1]]
; CHECK: qvfcmpeq [[REG4:[0-9]+]], [[REG3]], [[REG2]]
; CHECK: qvfsel 1, [[REG4]], 1, 2
; CHECK: blr
}

define <4 x i1> @test3(<4 x i1> %a) nounwind readnone {
entry:
  %v = and <4 x i1> %a, <i1 0, i1 undef, i1 1, i1 1>
  ret <4 x i1> %v

; CHECK-LABEL: @test3
; CHECK: qvlfsx [[REG:[0-9]+]],
; qvflogical 1, 1, [[REG]], 1
; blr
}

define <4 x i1> @test4(<4 x i1> %a, <4 x i1>* %t) nounwind {
entry:
  %q = load <4 x i1>, <4 x i1>* %t, align 16
  %v = and <4 x i1> %a, %q
  ret <4 x i1> %v

; CHECK-LABEL: @test4
; CHECK-DAG: lbz
; CHECK-DAG: qvlfdx [[REG1:[0-9]+]],
; CHECK-DAG: stw
; CHECK-DAG: qvlfiwzx [[REG2:[0-9]+]],
; CHECK-DAG: qvfcfidu [[REG3:[0-9]+]], [[REG2]]
; CHECK: qvfcmpeq [[REG4:[0-9]+]], [[REG3]], [[REG1]]
; CHECK: qvflogical 1, 1, [[REG4]], 1
; CHECK: blr
}

define void @test5(<4 x i1> %a) nounwind {
entry:
  store <4 x i1> %a, <4 x i1>* @R
  ret void

; CHECK-LABEL: @test5
; CHECK: qvlfdx [[REG1:[0-9]+]],
; CHECK: qvfmadd [[REG2:[0-9]+]], 1, [[REG1]], [[REG1]]
; CHECK: qvfctiwu [[REG3:[0-9]+]], [[REG2]]
; CHECK: qvstfiwx [[REG3]],
; CHECK: lwz
; CHECK: stb
; CHECK: blr
}

define i1 @test6(<4 x i1> %a) nounwind {
entry:
  %r = extractelement <4 x i1> %a, i32 2
  ret i1 %r

; CHECK-LABEL: @test6
; CHECK: qvlfdx [[REG1:[0-9]+]],
; CHECK: qvfmadd [[REG2:[0-9]+]], 1, [[REG1]], [[REG1]]
; CHECK: qvfctiwu [[REG3:[0-9]+]], [[REG2]]
; CHECK: qvstfiwx [[REG3]],
; CHECK: lwz
; CHECK: blr
}

define i1 @test7(<4 x i1> %a) nounwind {
entry:
  %r = extractelement <4 x i1> %a, i32 2
  %s = extractelement <4 x i1> %a, i32 3
  %q = and i1 %r, %s
  ret i1 %q

; CHECK-LABEL: @test7
; CHECK: qvlfdx [[REG1:[0-9]+]],
; CHECK: qvfmadd [[REG2:[0-9]+]], 1, [[REG1]], [[REG1]]
; CHECK: qvfctiwu [[REG3:[0-9]+]], [[REG2]]
; CHECK: qvstfiwx [[REG3]],
; CHECK-DAG: lwz [[REG4:[0-9]+]],
; FIXME: We're storing the vector twice, and that's silly.
; CHECK-DAG: qvstfiwx [[REG3]],
; CHECK: lwz [[REG5:[0-9]+]],
; CHECK: and 3,
; CHECK: blr
}

define i1 @test8(<3 x i1> %a) nounwind {
entry:
  %r = extractelement <3 x i1> %a, i32 2
  ret i1 %r

; CHECK-LABEL: @test8
; CHECK: qvlfdx [[REG1:[0-9]+]],
; CHECK: qvfmadd [[REG2:[0-9]+]], 1, [[REG1]], [[REG1]]
; CHECK: qvfctiwu [[REG3:[0-9]+]], [[REG2]]
; CHECK: qvstfiwx [[REG3]],
; CHECK: lwz
; CHECK: blr
}

define <3 x float> @test9(<3 x float> %a, <3 x float> %b, i1 %c1, i1 %c2, i1 %c3) nounwind readnone {
entry:
  %v = insertelement <3 x i1> undef, i1 %c1, i32 0
  %v2 = insertelement <3 x i1> %v, i1 %c2, i32 1
  %v3 = insertelement <3 x i1> %v2, i1 %c3, i32 2
  %r = select <3 x i1> %v3, <3 x float> %a, <3 x float> %b
  ret <3 x float> %r

; CHECK-LABEL: @test9
; CHECK: stw
; CHECK-DAG: qvlfiwzx [[REG1:[0-9]+]],
; CHECK-DAG: qvlfdx [[REG2:[0-9]+]],
; CHECK-DAG: qvfcfidu [[REG3:[0-9]+]], [[REG1]]
; CHECK: qvfcmpeq [[REG4:[0-9]+]], [[REG3]], [[REG2]]
; CHECK: qvfsel 1, [[REG4]], 1, 2
; CHECK: blr
}

