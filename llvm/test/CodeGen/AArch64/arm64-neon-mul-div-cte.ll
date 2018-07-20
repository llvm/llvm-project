; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon | FileCheck %s

define <16 x i8> @div16xi8(<16 x i8> %x) {
; CHECK-LABEL: div16xi8:
; CHECK:       movi   [[DIVISOR:(v[0-9]+)]].16b, #41
; CHECK-NEXT:  smull2 [[SMULL2:(v[0-9]+)]].8h, v0.16b, [[DIVISOR]].16b
; CHECK-NEXT:  smull  [[SMULL:(v[0-9]+)]].8h, v0.8b, [[DIVISOR]].8b
; CHECK-NEXT:  uzp2   [[UZP2:(v[0-9]+).16b]], [[SMULL]].16b, [[SMULL2]].16b
; CHECK-NEXT:  sshr   [[SSHR:(v[0-9]+.16b)]], [[UZP2]], #2
; CHECK-NEXT:  usra   v0.16b, [[SSHR]], #7
  %div = sdiv <16 x i8> %x, <i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25, i8 25>
  ret <16 x i8> %div
}

define <8 x i16> @div8xi16(<8 x i16> %x) {
; CHECK-LABEL: div8xi16:
; CHECK:       mov    [[TMP:(w[0-9]+)]], #40815
; CHECK-NEXT:  dup    [[DIVISOR:(v[0-9]+)]].8h, [[TMP]]
; CHECK-NEXT:  smull2 [[SMULL2:(v[0-9]+)]].4s, v0.8h, [[DIVISOR]].8h
; CHECK-NEXT:  smull  [[SMULL:(v[0-9]+)]].4s, v0.4h, [[DIVISOR]].4h
; CHECK-NEXT:  uzp2   [[UZP2:(v[0-9]+).8h]], [[SMULL]].8h, [[SMULL2]].8h
; CHECK-NEXT:  add    [[ADD:(v[0-9]+).8h]], [[UZP2]], v0.8h
; CHECK-NEXT:  sshr   [[SSHR:(v[0-9]+).8h]], [[ADD]], #12
; CHECK-NEXT:  usra   v0.8h, [[SSHR]], #15
  %div = sdiv <8 x i16> %x, <i16 6577, i16 6577, i16 6577, i16 6577, i16 6577, i16 6577, i16 6577, i16 6577>
  ret <8 x i16> %div
}

define <4 x i32> @div32xi4(<4 x i32> %x) {
; CHECK-LABEL: div32xi4:
; CHECK:       mov    [[TMP:(w[0-9]+)]], #7527
; CHECK-NEXT:  movk   [[TMP]], #28805, lsl #16
; CHECK-NEXT:  dup    [[DIVISOR:(v[0-9]+)]].4s, [[TMP]]
; CHECK-NEXT:  smull2 [[SMULL2:(v[0-9]+)]].2d, v0.4s, [[DIVISOR]].4s
; CHECK-NEXT:  smull  [[SMULL:(v[0-9]+)]].2d, v0.2s, [[DIVISOR]].2s
; CHECK-NEXT:  uzp2   [[UZP2:(v[0-9]+).4s]], [[SMULL]].4s, [[SMULL2]].4s
; CHECK-NEXT:  sshr   [[SSHR:(v[0-9]+.4s)]], [[UZP2]], #22
; CHECK-NEXT:  usra   v0.4s, [[UZP2]], #31
  %div = sdiv <4 x i32> %x, <i32 9542677, i32 9542677, i32 9542677, i32 9542677>
  ret <4 x i32> %div
}

define <16 x i8> @udiv16xi8(<16 x i8> %x) {
; CHECK-LABEL: udiv16xi8:
; CHECK:       movi   [[DIVISOR:(v[0-9]+)]].16b, #121
; CHECK-NEXT:  umull2 [[UMULL2:(v[0-9]+)]].8h, v0.16b, [[DIVISOR]].16b
; CHECK-NEXT:  umull  [[UMULL:(v[0-9]+)]].8h, v0.8b, [[DIVISOR]].8b
; CHECK-NEXT:  uzp2   [[UZP2:(v[0-9]+).16b]], [[UMULL]].16b, [[UMULL2]].16b
; CHECK-NEXT:  ushr   v0.16b, [[UZP2]], #5
  %div = udiv <16 x i8> %x, <i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68, i8 68>
  ret <16 x i8> %div
}

define <8 x i16> @udiv8xi16(<8 x i16> %x) {
; CHECK-LABEL: udiv8xi16:
; CHECK:       mov    [[TMP:(w[0-9]+)]], #16593
; CHECK-NEXT:  dup    [[DIVISOR:(v[0-9]+)]].8h, [[TMP]]
; CHECK-NEXT:  umull2 [[UMULL2:(v[0-9]+)]].4s, v0.8h, [[DIVISOR]].8h
; CHECK-NEXT:  umull  [[UMULL:(v[0-9]+)]].4s, v0.4h, [[DIVISOR]].4h
; CHECK-NEXT:  uzp2   [[UZP2:(v[0-9]+).8h]], [[UMULL]].8h, [[SMULL2]].8h
; CHECK-NEXT:  sub    [[SUB:(v[0-9]+).8h]], v0.8h, [[UZP2]]
; CHECK-NEXT:  usra   [[USRA:(v[0-9]+).8h]], [[SUB]], #1
; CHECK-NEXT:  ushr   v0.8h, [[USRA]], #12
  %div = udiv <8 x i16> %x, <i16 6537, i16 6537, i16 6537, i16 6537, i16 6537, i16 6537, i16 6537, i16 6537>
  ret <8 x i16> %div
}

define <4 x i32> @udiv32xi4(<4 x i32> %x) {
; CHECK-LABEL: udiv32xi4:
; CHECK:       mov    [[TMP:(w[0-9]+)]], #16747
; CHECK-NEXT:  movk   [[TMP]], #31439, lsl #16
; CHECK-NEXT:  dup    [[DIVISOR:(v[0-9]+)]].4s, [[TMP]]
; CHECK-NEXT:  umull2 [[UMULL2:(v[0-9]+)]].2d, v0.4s, [[DIVISOR]].4s
; CHECK-NEXT:  umull  [[UMULL:(v[0-9]+)]].2d, v0.2s, [[DIVISOR]].2s
; CHECK-NEXT:  uzp2   [[UZP2:(v[0-9]+).4s]], [[UMULL]].4s, [[SMULL2]].4s
; CHECK-NEXT:  ushr   v0.4s, [[UZP2]], #22
  %div = udiv <4 x i32> %x, <i32 8743143, i32 8743143, i32 8743143, i32 8743143>
  ret <4 x i32> %div
}
