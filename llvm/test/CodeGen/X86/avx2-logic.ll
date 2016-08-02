; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core-avx2 -mattr=+avx2 | FileCheck %s

; CHECK: vpandn
; CHECK: vpandn  %ymm
; CHECK: ret
define <4 x i64> @vpandn(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %y = xor <4 x i64> %a2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %x = and <4 x i64> %a, %y
  ret <4 x i64> %x
}

; CHECK: vpand
; CHECK: vpand %ymm
; CHECK: ret
define <4 x i64> @vpand(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = and <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK: vpor
; CHECK: vpor %ymm
; CHECK: ret
define <4 x i64> @vpor(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = or <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK: vpxor
; CHECK: vpxor %ymm
; CHECK: ret
define <4 x i64> @vpxor(<4 x i64> %a, <4 x i64> %b) nounwind uwtable readnone ssp {
entry:
  ; Force the execution domain with an add.
  %a2 = add <4 x i64> %a, <i64 1, i64 1, i64 1, i64 1>
  %x = xor <4 x i64> %a2, %b
  ret <4 x i64> %x
}

; CHECK: vpblendvb
; CHECK: vpblendvb %ymm
; CHECK: ret
define <32 x i8> @vpblendvb(<32 x i1> %cond, <32 x i8> %x, <32 x i8> %y) {
  %min = select <32 x i1> %cond, <32 x i8> %x, <32 x i8> %y
  ret <32 x i8> %min
}

define <8 x i32> @signd(<8 x i32> %a, <8 x i32> %b) nounwind {
entry:
; CHECK-LABEL: signd:
; CHECK: psignd
; CHECK-NOT: sub
; CHECK: ret
  %b.lobit = ashr <8 x i32> %b, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  %sub = sub nsw <8 x i32> zeroinitializer, %a
  %0 = xor <8 x i32> %b.lobit, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %1 = and <8 x i32> %a, %0
  %2 = and <8 x i32> %b.lobit, %sub
  %cond = or <8 x i32> %1, %2
  ret <8 x i32> %cond
}

define <8 x i32> @blendvb(<8 x i32> %b, <8 x i32> %a, <8 x i32> %c) nounwind {
entry:
; CHECK-LABEL: blendvb:
; CHECK: pblendvb
; CHECK: ret
  %b.lobit = ashr <8 x i32> %b, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  %sub = sub nsw <8 x i32> zeroinitializer, %a
  %0 = xor <8 x i32> %b.lobit, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %1 = and <8 x i32> %c, %0
  %2 = and <8 x i32> %a, %b.lobit
  %cond = or <8 x i32> %1, %2
  ret <8 x i32> %cond
}

define <8 x i32> @allOnes() nounwind {
; CHECK: vpcmpeqd
; CHECK-NOT: vinsert
        ret <8 x i32> <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
}

define <16 x i16> @allOnes2() nounwind {
; CHECK: vpcmpeqd
; CHECK-NOT: vinsert
        ret <16 x i16> <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
}
