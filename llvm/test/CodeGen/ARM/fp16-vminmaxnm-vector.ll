; RUN: llc < %s -mtriple=arm-eabi -mattr=+v8.2a,+neon,+fullfp16 -float-abi=hard | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7a -mattr=+v8.2a,+neon,+fullfp16 -float-abi=hard | FileCheck %s

; 4-element vector

; Ordered

define <4 x half> @test1(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test1:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ogt <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test2(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test2:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ogt <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

define <4 x half> @test3(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test3:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast oge <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

define <4 x half> @test4(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test4:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast oge <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test5(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test5:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast olt <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test6(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test6:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast olt <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

define <4 x half> @test7(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test7:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ole <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test8(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test8:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ole <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

; Unordered

define <4 x half> @test11(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test11:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ugt <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test12(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test12:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ugt <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

define <4 x half> @test13(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test13:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast uge <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

define <4 x half> @test14(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test14:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast uge <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test15(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test15:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ult <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test16(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test16:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ult <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

define <4 x half> @test17(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test17:
; CHECK:         vminnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ule <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %A, <4 x half> %B
  ret <4 x half> %tmp4
}

define <4 x half> @test18(<4 x half> %A, <4 x half> %B) {
; CHECK-LABEL: test18:
; CHECK:         vmaxnm.f16 d0, d0, d1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ule <4 x half> %A, %B
  %tmp4 = select <4 x i1> %tmp3, <4 x half> %B, <4 x half> %A
  ret <4 x half> %tmp4
}

; 8-element vector

; Ordered

define <8 x half> @test201(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test201:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ogt <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test202(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test202:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ogt <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}

define <8 x half> @test203(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test203:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast oge <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test204(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test204:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast oge <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}

define <8 x half> @test205(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test205:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast olt <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test206(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test206:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast olt <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}

define <8 x half> @test207(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test207:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ole <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test208(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test208:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ole <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}

; Unordered

define <8 x half> @test209(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test209:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ugt <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test210(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test210:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ugt <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}

define <8 x half> @test211(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test211:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast uge <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test214(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test214:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast uge <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}

define <8 x half> @test215(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test215:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ult <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test216(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test216:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ult <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}

define <8 x half> @test217(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test217:
; CHECK:         vminnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ule <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %A, <8 x half> %B
  ret <8 x half> %tmp4
}

define <8 x half> @test218(<8 x half> %A, <8 x half> %B) {
; CHECK-LABEL: test218:
; CHECK:         vmaxnm.f16 q0, q0, q1
; CHECK-NEXT:    bx lr
  %tmp3 = fcmp fast ule <8 x half> %A, %B
  %tmp4 = select <8 x i1> %tmp3, <8 x half> %B, <8 x half> %A
  ret <8 x half> %tmp4
}
