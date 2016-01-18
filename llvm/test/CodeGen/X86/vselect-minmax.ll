; RUN: llc -march=x86-64 -mcpu=core2 < %s | FileCheck %s -check-prefix=SSE2
; RUN: llc -march=x86-64 -mcpu=corei7 < %s | FileCheck %s -check-prefix=SSE4
; RUN: llc -march=x86-64 -mcpu=corei7-avx < %s | FileCheck %s -check-prefix=AVX1
; RUN: llc -march=x86-64 -mcpu=core-avx2 -mattr=+avx2 < %s | FileCheck %s -check-prefix=AVX2
; RUN: llc -march=x86-64 -mcpu=knl < %s | FileCheck %s  -check-prefix=AVX2 -check-prefix=AVX512F
; RUN: llc -march=x86-64 -mcpu=skx < %s | FileCheck %s  -check-prefix=AVX512BW -check-prefix=AVX512VL -check-prefix=AVX512F

define <16 x i8> @test1(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp slt <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE4-LABEL: test1:
; SSE4: pminsb

; AVX1-LABEL: test1:
; AVX1: vpminsb

; AVX2-LABEL: test1:
; AVX2: vpminsb

; AVX512VL-LABEL: test1:
; AVX512VL: vpminsb
}

define <16 x i8> @test2(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp sle <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE4-LABEL: test2:
; SSE4: pminsb

; AVX1-LABEL: test2:
; AVX1: vpminsb

; AVX2-LABEL: test2:
; AVX2: vpminsb

; AVX512VL-LABEL: test2:
; AVX512VL: vpminsb
}

define <16 x i8> @test3(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp sgt <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE4-LABEL: test3:
; SSE4: pmaxsb

; AVX1-LABEL: test3:
; AVX1: vpmaxsb

; AVX2-LABEL: test3:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test3:
; AVX512VL: vpmaxsb
}

define <16 x i8> @test4(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp sge <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE4-LABEL: test4:
; SSE4: pmaxsb

; AVX1-LABEL: test4:
; AVX1: vpmaxsb

; AVX2-LABEL: test4:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test4:
; AVX512VL: vpmaxsb
}

define <16 x i8> @test5(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp ult <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE2-LABEL: test5:
; SSE2: pminub

; AVX1-LABEL: test5:
; AVX1: vpminub

; AVX2-LABEL: test5:
; AVX2: vpminub

; AVX512VL-LABEL: test5:
; AVX512VL: vpminub 
}

define <16 x i8> @test6(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp ule <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE2-LABEL: test6:
; SSE2: pminub

; AVX1-LABEL: test6:
; AVX1: vpminub

; AVX2-LABEL: test6:
; AVX2: vpminub

; AVX512VL-LABEL: test6:
; AVX512VL: vpminub
}

define <16 x i8> @test7(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp ugt <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE2-LABEL: test7:
; SSE2: pmaxub

; AVX1-LABEL: test7:
; AVX1: vpmaxub

; AVX2-LABEL: test7:
; AVX2: vpmaxub

; AVX512VL-LABEL: test7:
; AVX512VL: vpmaxub
}

define <16 x i8> @test8(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp uge <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel

; SSE2-LABEL: test8:
; SSE2: pmaxub

; AVX1-LABEL: test8:
; AVX1: vpmaxub

; AVX2-LABEL: test8:
; AVX2: vpmaxub

; AVX512VL-LABEL: test8:
; AVX512VL: vpmaxub
}

define <8 x i16> @test9(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp slt <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE2-LABEL: test9:
; SSE2: pminsw

; AVX1-LABEL: test9:
; AVX1: vpminsw

; AVX2-LABEL: test9:
; AVX2: vpminsw

; AVX512VL-LABEL: test9:
; AVX512VL: vpminsw 
}

define <8 x i16> @test10(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp sle <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE2-LABEL: test10:
; SSE2: pminsw

; AVX1-LABEL: test10:
; AVX1: vpminsw

; AVX2-LABEL: test10:
; AVX2: vpminsw

; AVX512VL-LABEL: test10:
; AVX512VL: vpminsw
}

define <8 x i16> @test11(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp sgt <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE2-LABEL: test11:
; SSE2: pmaxsw

; AVX1-LABEL: test11:
; AVX1: vpmaxsw

; AVX2-LABEL: test11:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test11:
; AVX512VL: vpmaxsw
}

define <8 x i16> @test12(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp sge <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE2-LABEL: test12:
; SSE2: pmaxsw

; AVX1-LABEL: test12:
; AVX1: vpmaxsw

; AVX2-LABEL: test12:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test12:
; AVX512VL: vpmaxsw
}

define <8 x i16> @test13(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp ult <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE4-LABEL: test13:
; SSE4: pminuw

; AVX1-LABEL: test13:
; AVX1: vpminuw

; AVX2-LABEL: test13:
; AVX2: vpminuw

; AVX512VL-LABEL: test13:
; AVX512VL: vpminuw
}

define <8 x i16> @test14(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp ule <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE4-LABEL: test14:
; SSE4: pminuw

; AVX1-LABEL: test14:
; AVX1: vpminuw

; AVX2-LABEL: test14:
; AVX2: vpminuw

; AVX512VL-LABEL: test14:
; AVX512VL: vpminuw 
}

define <8 x i16> @test15(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp ugt <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE4-LABEL: test15:
; SSE4: pmaxuw

; AVX1-LABEL: test15:
; AVX1: vpmaxuw

; AVX2-LABEL: test15:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test15:
; AVX512VL: vpmaxuw
}

define <8 x i16> @test16(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp uge <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %a, <8 x i16> %b
  ret <8 x i16> %sel

; SSE4-LABEL: test16:
; SSE4: pmaxuw

; AVX1-LABEL: test16:
; AVX1: vpmaxuw

; AVX2-LABEL: test16:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test16:
; AVX512VL: vpmaxuw
}

define <4 x i32> @test17(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp slt <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test17:
; SSE4: pminsd

; AVX1-LABEL: test17:
; AVX1: vpminsd

; AVX2-LABEL: test17:
; AVX2: vpminsd

; AVX512VL-LABEL: test17:
; AVX512VL: vpminsd
}

define <4 x i32> @test18(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp sle <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test18:
; SSE4: pminsd

; AVX1-LABEL: test18:
; AVX1: vpminsd

; AVX2-LABEL: test18:
; AVX2: vpminsd

; AVX512VL-LABEL: test18:
; AVX512VL: vpminsd
}

define <4 x i32> @test19(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp sgt <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test19:
; SSE4: pmaxsd

; AVX1-LABEL: test19:
; AVX1: vpmaxsd

; AVX2-LABEL: test19:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test19:
; AVX512VL: vpmaxsd
}

define <4 x i32> @test20(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp sge <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test20:
; SSE4: pmaxsd

; AVX1-LABEL: test20:
; AVX1: vpmaxsd

; AVX2-LABEL: test20:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test20:
; AVX512VL: vpmaxsd
}

define <4 x i32> @test21(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp ult <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test21:
; SSE4: pminud

; AVX1-LABEL: test21:
; AVX1: vpminud

; AVX2-LABEL: test21:
; AVX2: vpminud

; AVX512VL-LABEL: test21:
; AVX512VL: vpminud
}

define <4 x i32> @test22(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp ule <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test22:
; SSE4: pminud

; AVX1-LABEL: test22:
; AVX1: vpminud

; AVX2-LABEL: test22:
; AVX2: vpminud

; AVX512VL-LABEL: test22:
; AVX512VL: vpminud
}

define <4 x i32> @test23(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp ugt <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test23:
; SSE4: pmaxud

; AVX1-LABEL: test23:
; AVX1: vpmaxud

; AVX2-LABEL: test23:
; AVX2: vpmaxud

; AVX512VL-LABEL: test23:
; AVX512VL: vpmaxud
}

define <4 x i32> @test24(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp uge <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel

; SSE4-LABEL: test24:
; SSE4: pmaxud

; AVX1-LABEL: test24:
; AVX1: vpmaxud

; AVX2-LABEL: test24:
; AVX2: vpmaxud

; AVX512VL-LABEL: test24:
; AVX512VL: vpmaxud
}

define <32 x i8> @test25(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp slt <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test25:
; AVX2: vpminsb

; AVX512VL-LABEL: test25:
; AVX512VL: vpminsb
}

define <32 x i8> @test26(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp sle <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test26:
; AVX2: vpminsb

; AVX512VL-LABEL: test26:
; AVX512VL: vpminsb
}

define <32 x i8> @test27(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp sgt <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test27:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test27:
; AVX512VL: vpmaxsb
}

define <32 x i8> @test28(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp sge <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test28:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test28:
; AVX512VL: vpmaxsb
}

define <32 x i8> @test29(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp ult <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test29:
; AVX2: vpminub

; AVX512VL-LABEL: test29:
; AVX512VL: vpminub
}

define <32 x i8> @test30(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp ule <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test30:
; AVX2: vpminub

; AVX512VL-LABEL: test30:
; AVX512VL: vpminub
}

define <32 x i8> @test31(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp ugt <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test31:
; AVX2: vpmaxub

; AVX512VL-LABEL: test31:
; AVX512VL: vpmaxub
}

define <32 x i8> @test32(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp uge <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel

; AVX2-LABEL: test32:
; AVX2: vpmaxub

; AVX512VL-LABEL: test32:
; AVX512VL: vpmaxub
}

define <16 x i16> @test33(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp slt <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test33:
; AVX2: vpminsw

; AVX512VL-LABEL: test33:
; AVX512VL: vpminsw 
}

define <16 x i16> @test34(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp sle <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test34:
; AVX2: vpminsw

; AVX512VL-LABEL: test34:
; AVX512VL: vpminsw
}

define <16 x i16> @test35(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp sgt <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test35:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test35:
; AVX512VL: vpmaxsw
}

define <16 x i16> @test36(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp sge <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test36:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test36:
; AVX512VL: vpmaxsw
}

define <16 x i16> @test37(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp ult <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test37:
; AVX2: vpminuw

; AVX512VL-LABEL: test37:
; AVX512VL: vpminuw
}

define <16 x i16> @test38(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp ule <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test38:
; AVX2: vpminuw

; AVX512VL-LABEL: test38:
; AVX512VL: vpminuw
}

define <16 x i16> @test39(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp ugt <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test39:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test39:
; AVX512VL: vpmaxuw
}

define <16 x i16> @test40(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp uge <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel

; AVX2-LABEL: test40:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test40:
; AVX512VL: vpmaxuw
}

define <8 x i32> @test41(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp slt <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test41:
; AVX2: vpminsd

; AVX512VL-LABEL: test41:
; AVX512VL: vpminsd
}

define <8 x i32> @test42(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp sle <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test42:
; AVX2: vpminsd

; AVX512VL-LABEL: test42:
; AVX512VL: vpminsd
}

define <8 x i32> @test43(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp sgt <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test43:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test43:
; AVX512VL: vpmaxsd
}

define <8 x i32> @test44(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp sge <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test44:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test44:
; AVX512VL: vpmaxsd
}

define <8 x i32> @test45(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp ult <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test45:
; AVX2: vpminud

; AVX512VL-LABEL: test45:
; AVX512VL: vpminud
}

define <8 x i32> @test46(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp ule <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test46:
; AVX2: vpminud

; AVX512VL-LABEL: test46:
; AVX512VL: vpminud
}

define <8 x i32> @test47(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp ugt <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test47:
; AVX2: vpmaxud

; AVX512VL-LABEL: test47:
; AVX512VL: vpmaxud
}

define <8 x i32> @test48(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp uge <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel

; AVX2-LABEL: test48:
; AVX2: vpmaxud

; AVX512VL-LABEL: test48:
; AVX512VL: vpmaxud
}

define <16 x i8> @test49(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp slt <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE4-LABEL: test49:
; SSE4: pmaxsb

; AVX1-LABEL: test49:
; AVX1: vpmaxsb

; AVX2-LABEL: test49:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test49:
; AVX512VL: vpmaxsb
}

define <16 x i8> @test50(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp sle <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE4-LABEL: test50:
; SSE4: pmaxsb

; AVX1-LABEL: test50:
; AVX1: vpmaxsb

; AVX2-LABEL: test50:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test50:
; AVX512VL: vpmaxsb
}

define <16 x i8> @test51(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp sgt <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE4-LABEL: test51:
; SSE4: pminsb

; AVX1-LABEL: test51:
; AVX1: vpminsb

; AVX2-LABEL: test51:
; AVX2: vpminsb

; AVX512VL-LABEL: test51:
; AVX512VL: vpminsb
}

define <16 x i8> @test52(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp sge <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE4-LABEL: test52:
; SSE4: pminsb

; AVX1-LABEL: test52:
; AVX1: vpminsb

; AVX2-LABEL: test52:
; AVX2: vpminsb

; AVX512VL-LABEL: test52:
; AVX512VL: vpminsb
}

define <16 x i8> @test53(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp ult <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE2-LABEL: test53:
; SSE2: pmaxub

; AVX1-LABEL: test53:
; AVX1: vpmaxub

; AVX2-LABEL: test53:
; AVX2: vpmaxub

; AVX512VL-LABEL: test53:
; AVX512VL: vpmaxub
}

define <16 x i8> @test54(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp ule <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE2-LABEL: test54:
; SSE2: pmaxub

; AVX1-LABEL: test54:
; AVX1: vpmaxub

; AVX2-LABEL: test54:
; AVX2: vpmaxub

; AVX512VL-LABEL: test54:
; AVX512VL: vpmaxub
}

define <16 x i8> @test55(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp ugt <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE2-LABEL: test55:
; SSE2: pminub

; AVX1-LABEL: test55:
; AVX1: vpminub

; AVX2-LABEL: test55:
; AVX2: vpminub

; AVX512VL-LABEL: test55:
; AVX512VL: vpminub
}

define <16 x i8> @test56(<16 x i8> %a, <16 x i8> %b) {
entry:
  %cmp = icmp uge <16 x i8> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i8> %b, <16 x i8> %a
  ret <16 x i8> %sel

; SSE2-LABEL: test56:
; SSE2: pminub

; AVX1-LABEL: test56:
; AVX1: vpminub

; AVX2-LABEL: test56:
; AVX2: vpminub

; AVX512VL-LABEL: test56:
; AVX512VL: vpminub
}

define <8 x i16> @test57(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp slt <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE2-LABEL: test57:
; SSE2: pmaxsw

; AVX1-LABEL: test57:
; AVX1: vpmaxsw

; AVX2-LABEL: test57:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test57:
; AVX512VL: vpmaxsw
}

define <8 x i16> @test58(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp sle <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE2-LABEL: test58:
; SSE2: pmaxsw

; AVX1-LABEL: test58:
; AVX1: vpmaxsw

; AVX2-LABEL: test58:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test58:
; AVX512VL: vpmaxsw
}

define <8 x i16> @test59(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp sgt <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE2-LABEL: test59:
; SSE2: pminsw

; AVX1-LABEL: test59:
; AVX1: vpminsw

; AVX2-LABEL: test59:
; AVX2: vpminsw

; AVX512VL-LABEL: test59:
; AVX512VL: vpminsw
}

define <8 x i16> @test60(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp sge <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE2-LABEL: test60:
; SSE2: pminsw

; AVX1-LABEL: test60:
; AVX1: vpminsw

; AVX2-LABEL: test60:
; AVX2: vpminsw

; AVX512VL-LABEL: test60:
; AVX512VL: vpminsw
}

define <8 x i16> @test61(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp ult <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE4-LABEL: test61:
; SSE4: pmaxuw

; AVX1-LABEL: test61:
; AVX1: vpmaxuw

; AVX2-LABEL: test61:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test61:
; AVX512VL: vpmaxuw
}

define <8 x i16> @test62(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp ule <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE4-LABEL: test62:
; SSE4: pmaxuw

; AVX1-LABEL: test62:
; AVX1: vpmaxuw

; AVX2-LABEL: test62:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test62:
; AVX512VL: vpmaxuw
}

define <8 x i16> @test63(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp ugt <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE4-LABEL: test63:
; SSE4: pminuw

; AVX1-LABEL: test63:
; AVX1: vpminuw

; AVX2-LABEL: test63:
; AVX2: vpminuw

; AVX512VL-LABEL: test63:
; AVX512VL: vpminuw
}

define <8 x i16> @test64(<8 x i16> %a, <8 x i16> %b) {
entry:
  %cmp = icmp uge <8 x i16> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i16> %b, <8 x i16> %a
  ret <8 x i16> %sel

; SSE4-LABEL: test64:
; SSE4: pminuw

; AVX1-LABEL: test64:
; AVX1: vpminuw

; AVX2-LABEL: test64:
; AVX2: vpminuw

; AVX512VL-LABEL: test64:
; AVX512VL: vpminuw
}

define <4 x i32> @test65(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp slt <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test65:
; SSE4: pmaxsd

; AVX1-LABEL: test65:
; AVX1: vpmaxsd

; AVX2-LABEL: test65:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test65:
; AVX512VL: vpmaxsd
}

define <4 x i32> @test66(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp sle <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test66:
; SSE4: pmaxsd

; AVX1-LABEL: test66:
; AVX1: vpmaxsd

; AVX2-LABEL: test66:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test66:
; AVX512VL: vpmaxsd
}

define <4 x i32> @test67(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp sgt <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test67:
; SSE4: pminsd

; AVX1-LABEL: test67:
; AVX1: vpminsd

; AVX2-LABEL: test67:
; AVX2: vpminsd

; AVX512VL-LABEL: test67:
; AVX512VL: vpminsd
}

define <4 x i32> @test68(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp sge <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test68:
; SSE4: pminsd

; AVX1-LABEL: test68:
; AVX1: vpminsd

; AVX2-LABEL: test68:
; AVX2: vpminsd

; AVX512VL-LABEL: test68:
; AVX512VL: vpminsd
}

define <4 x i32> @test69(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp ult <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test69:
; SSE4: pmaxud

; AVX1-LABEL: test69:
; AVX1: vpmaxud

; AVX2-LABEL: test69:
; AVX2: vpmaxud

; AVX512VL-LABEL: test69:
; AVX512VL: vpmaxud
}

define <4 x i32> @test70(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp ule <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test70:
; SSE4: pmaxud

; AVX1-LABEL: test70:
; AVX1: vpmaxud

; AVX2-LABEL: test70:
; AVX2: vpmaxud

; AVX512VL-LABEL: test70:
; AVX512VL: vpmaxud
}

define <4 x i32> @test71(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp ugt <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test71:
; SSE4: pminud

; AVX1-LABEL: test71:
; AVX1: vpminud

; AVX2-LABEL: test71:
; AVX2: vpminud

; AVX512VL-LABEL: test71:
; AVX512VL: vpminud
}

define <4 x i32> @test72(<4 x i32> %a, <4 x i32> %b) {
entry:
  %cmp = icmp uge <4 x i32> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i32> %b, <4 x i32> %a
  ret <4 x i32> %sel

; SSE4-LABEL: test72:
; SSE4: pminud

; AVX1-LABEL: test72:
; AVX1: vpminud

; AVX2-LABEL: test72:
; AVX2: vpminud

; AVX512VL-LABEL: test72:
; AVX512VL: vpminud
}

define <32 x i8> @test73(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp slt <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test73:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test73:
; AVX512VL: vpmaxsb
}

define <32 x i8> @test74(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp sle <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test74:
; AVX2: vpmaxsb

; AVX512VL-LABEL: test74:
; AVX512VL: vpmaxsb 
}

define <32 x i8> @test75(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp sgt <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test75:
; AVX2: vpminsb

; AVX512VL-LABEL: test75:
; AVX512VL: vpminsb
}

define <32 x i8> @test76(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp sge <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test76:
; AVX2: vpminsb

; AVX512VL-LABEL: test76:
; AVX512VL: vpminsb
}

define <32 x i8> @test77(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp ult <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test77:
; AVX2: vpmaxub

; AVX512VL-LABEL: test77:
; AVX512VL: vpmaxub
}

define <32 x i8> @test78(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp ule <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test78:
; AVX2: vpmaxub

; AVX512VL-LABEL: test78:
; AVX512VL: vpmaxub
}

define <32 x i8> @test79(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp ugt <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test79:
; AVX2: vpminub

; AVX512VL-LABEL: test79:
; AVX512VL: vpminub
}

define <32 x i8> @test80(<32 x i8> %a, <32 x i8> %b) {
entry:
  %cmp = icmp uge <32 x i8> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i8> %b, <32 x i8> %a
  ret <32 x i8> %sel

; AVX2-LABEL: test80:
; AVX2: vpminub

; AVX512VL-LABEL: test80:
; AVX512VL: vpminub
}

define <16 x i16> @test81(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp slt <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test81:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test81:
; AVX512VL: vpmaxsw
}

define <16 x i16> @test82(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp sle <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test82:
; AVX2: vpmaxsw

; AVX512VL-LABEL: test82:
; AVX512VL: vpmaxsw
}

define <16 x i16> @test83(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp sgt <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test83:
; AVX2: vpminsw

; AVX512VL-LABEL: test83:
; AVX512VL: vpminsw
}

define <16 x i16> @test84(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp sge <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test84:
; AVX2: vpminsw

; AVX512VL-LABEL: test84:
; AVX512VL: vpminsw
}

define <16 x i16> @test85(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp ult <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test85:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test85:
; AVX512VL: vpmaxuw
}

define <16 x i16> @test86(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp ule <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test86:
; AVX2: vpmaxuw

; AVX512VL-LABEL: test86:
; AVX512VL: vpmaxuw
}

define <16 x i16> @test87(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp ugt <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test87:
; AVX2: vpminuw

; AVX512VL-LABEL: test87:
; AVX512VL: vpminuw
}

define <16 x i16> @test88(<16 x i16> %a, <16 x i16> %b) {
entry:
  %cmp = icmp uge <16 x i16> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i16> %b, <16 x i16> %a
  ret <16 x i16> %sel

; AVX2-LABEL: test88:
; AVX2: vpminuw

; AVX512VL-LABEL: test88:
; AVX512VL: vpminuw
}

define <8 x i32> @test89(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp slt <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test89:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test89:
; AVX512VL: vpmaxsd
}

define <8 x i32> @test90(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp sle <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test90:
; AVX2: vpmaxsd

; AVX512VL-LABEL: test90:
; AVX512VL: vpmaxsd
}

define <8 x i32> @test91(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp sgt <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test91:
; AVX2: vpminsd

; AVX512VL-LABEL: test91:
; AVX512VL: vpminsd
}

define <8 x i32> @test92(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp sge <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test92:
; AVX2: vpminsd

; AVX512VL-LABEL: test92:
; AVX512VL: vpminsd
}

define <8 x i32> @test93(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp ult <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test93:
; AVX2: vpmaxud

; AVX512VL-LABEL: test93:
; AVX512VL: vpmaxud
}

define <8 x i32> @test94(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp ule <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test94:
; AVX2: vpmaxud

; AVX512VL-LABEL: test94:
; AVX512VL: vpmaxud
}

define <8 x i32> @test95(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp ugt <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test95:
; AVX2: vpminud

; AVX512VL-LABEL: test95:
; AVX512VL: vpminud
}

define <8 x i32> @test96(<8 x i32> %a, <8 x i32> %b) {
entry:
  %cmp = icmp uge <8 x i32> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i32> %b, <8 x i32> %a
  ret <8 x i32> %sel

; AVX2-LABEL: test96:
; AVX2: vpminud

; AVX512VL-LABEL: test96:
; AVX512VL: vpminud
}

; ----------------------------

define <64 x i8> @test97(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp slt <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test97:
; AVX512BW: vpminsb {{.*}}
}

define <64 x i8> @test98(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp sle <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test98:
; AVX512BW: vpminsb {{.*}}
}

define <64 x i8> @test99(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp sgt <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test99:
; AVX512BW: vpmaxsb {{.*}}
}

define <64 x i8> @test100(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp sge <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test100:
; AVX512BW: vpmaxsb {{.*}}
}

define <64 x i8> @test101(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp ult <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test101:
; AVX512BW: vpminub {{.*}}
}

define <64 x i8> @test102(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp ule <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test102:
; AVX512BW: vpminub {{.*}}
}

define <64 x i8> @test103(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp ugt <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test103:
; AVX512BW: vpmaxub {{.*}}
}

define <64 x i8> @test104(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp uge <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %a, <64 x i8> %b
  ret <64 x i8> %sel

; AVX512BW-LABEL: test104:
; AVX512BW: vpmaxub {{.*}}
}

define <32 x i16> @test105(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp slt <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test105:
; AVX512BW: vpminsw {{.*}}
}

define <32 x i16> @test106(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp sle <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test106:
; AVX512BW: vpminsw {{.*}}
}

define <32 x i16> @test107(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp sgt <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test107:
; AVX512BW: vpmaxsw {{.*}}
}

define <32 x i16> @test108(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp sge <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test108:
; AVX512BW: vpmaxsw {{.*}}
}

define <32 x i16> @test109(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp ult <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test109:
; AVX512BW: vpminuw {{.*}}
}

define <32 x i16> @test110(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp ule <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test110:
; AVX512BW: vpminuw {{.*}}
}

define <32 x i16> @test111(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp ugt <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test111:
; AVX512BW: vpmaxuw {{.*}}
}

define <32 x i16> @test112(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp uge <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %a, <32 x i16> %b
  ret <32 x i16> %sel

; AVX512BW-LABEL: test112:
; AVX512BW: vpmaxuw {{.*}}
}

define <16 x i32> @test113(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp slt <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test113:
; AVX512F: vpminsd {{.*}}
}

define <16 x i32> @test114(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp sle <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test114:
; AVX512F: vpminsd {{.*}}
}

define <16 x i32> @test115(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp sgt <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test115:
; AVX512F: vpmaxsd {{.*}}
}

define <16 x i32> @test116(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp sge <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test116:
; AVX512F: vpmaxsd {{.*}}
}

define <16 x i32> @test117(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp ult <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test117:
; AVX512F: vpminud {{.*}}
}

define <16 x i32> @test118(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp ule <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test118:
; AVX512F: vpminud {{.*}}
}

define <16 x i32> @test119(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp ugt <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test119:
; AVX512F: vpmaxud {{.*}}
}

define <16 x i32> @test120(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp uge <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %a, <16 x i32> %b
  ret <16 x i32> %sel

; AVX512F-LABEL: test120:
; AVX512F: vpmaxud {{.*}}
}

define <8 x i64> @test121(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp slt <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test121:
; AVX512F: vpminsq {{.*}}
}

define <8 x i64> @test122(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp sle <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test122:
; AVX512F: vpminsq {{.*}}
}

define <8 x i64> @test123(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp sgt <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test123:
; AVX512F: vpmaxsq {{.*}}
}

define <8 x i64> @test124(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp sge <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test124:
; AVX512F: vpmaxsq {{.*}}
}

define <8 x i64> @test125(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp ult <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test125:
; AVX512F: vpminuq {{.*}}
}

define <8 x i64> @test126(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp ule <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test126:
; AVX512F: vpminuq {{.*}}
}

define <8 x i64> @test127(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp ugt <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test127:
; AVX512F: vpmaxuq {{.*}}
}

define <8 x i64> @test128(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp uge <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %a, <8 x i64> %b
  ret <8 x i64> %sel

; AVX512F-LABEL: test128:
; AVX512F: vpmaxuq {{.*}}
}

define <64 x i8> @test129(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp slt <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test129:
; AVX512BW: vpmaxsb
}

define <64 x i8> @test130(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp sle <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test130:
; AVX512BW: vpmaxsb
}

define <64 x i8> @test131(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp sgt <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test131:
; AVX512BW: vpminsb
}

define <64 x i8> @test132(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp sge <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test132:
; AVX512BW: vpminsb
}

define <64 x i8> @test133(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp ult <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test133:
; AVX512BW: vpmaxub
}

define <64 x i8> @test134(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp ule <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test134:
; AVX512BW: vpmaxub
}

define <64 x i8> @test135(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp ugt <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test135:
; AVX512BW: vpminub
}

define <64 x i8> @test136(<64 x i8> %a, <64 x i8> %b) {
entry:
  %cmp = icmp uge <64 x i8> %a, %b
  %sel = select <64 x i1> %cmp, <64 x i8> %b, <64 x i8> %a
  ret <64 x i8> %sel

; AVX512BW-LABEL: test136:
; AVX512BW: vpminub
}

define <32 x i16> @test137(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp slt <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test137:
; AVX512BW: vpmaxsw
}

define <32 x i16> @test138(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp sle <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test138:
; AVX512BW: vpmaxsw
}

define <32 x i16> @test139(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp sgt <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test139:
; AVX512BW: vpminsw
}

define <32 x i16> @test140(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp sge <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test140:
; AVX512BW: vpminsw
}

define <32 x i16> @test141(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp ult <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test141:
; AVX512BW: vpmaxuw
}

define <32 x i16> @test142(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp ule <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test142:
; AVX512BW: vpmaxuw
}

define <32 x i16> @test143(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp ugt <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test143:
; AVX512BW: vpminuw
}

define <32 x i16> @test144(<32 x i16> %a, <32 x i16> %b) {
entry:
  %cmp = icmp uge <32 x i16> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i16> %b, <32 x i16> %a
  ret <32 x i16> %sel

; AVX512BW-LABEL: test144:
; AVX512BW: vpminuw
}

define <16 x i32> @test145(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp slt <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test145:
; AVX512F: vpmaxsd
}

define <16 x i32> @test146(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp sle <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test146:
; AVX512F: vpmaxsd
}

define <16 x i32> @test147(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp sgt <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test147:
; AVX512F: vpminsd
}

define <16 x i32> @test148(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp sge <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test148:
; AVX512F: vpminsd
}

define <16 x i32> @test149(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp ult <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test149:
; AVX512F: vpmaxud
}

define <16 x i32> @test150(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp ule <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test150:
; AVX512F: vpmaxud
}

define <16 x i32> @test151(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp ugt <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test151:
; AVX512F: vpminud
}

define <16 x i32> @test152(<16 x i32> %a, <16 x i32> %b) {
entry:
  %cmp = icmp uge <16 x i32> %a, %b
  %sel = select <16 x i1> %cmp, <16 x i32> %b, <16 x i32> %a
  ret <16 x i32> %sel

; AVX512F-LABEL: test152:
; AVX512F: vpminud
}

; -----------------------

define <8 x i64> @test153(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp slt <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test153:
; AVX512F: vpmaxsq
}

define <8 x i64> @test154(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp sle <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test154:
; AVX512F: vpmaxsq
}

define <8 x i64> @test155(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp sgt <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test155:
; AVX512F: vpminsq
}

define <8 x i64> @test156(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp sge <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test156:
; AVX512F: vpminsq
}

define <8 x i64> @test157(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp ult <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test157:
; AVX512F: vpmaxuq
}

define <8 x i64> @test158(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp ule <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test158:
; AVX512F: vpmaxuq
}

define <8 x i64> @test159(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp ugt <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test159:
; AVX512F: vpminuq
}

define <8 x i64> @test160(<8 x i64> %a, <8 x i64> %b) {
entry:
  %cmp = icmp uge <8 x i64> %a, %b
  %sel = select <8 x i1> %cmp, <8 x i64> %b, <8 x i64> %a
  ret <8 x i64> %sel

; AVX512F-LABEL: test160:
; AVX512F: vpminuq
}

define <4 x i64> @test161(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp slt <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test161:
; AVX512VL: vpminsq
}

define <4 x i64> @test162(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp sle <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test162:
; AVX512VL: vpminsq
}

define <4 x i64> @test163(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp sgt <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test163:
; AVX512VL: vpmaxsq 
}

define <4 x i64> @test164(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp sge <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test164:
; AVX512VL: vpmaxsq
}

define <4 x i64> @test165(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp ult <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test165:
; AVX512VL: vpminuq 
}

define <4 x i64> @test166(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp ule <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test166:
; AVX512VL: vpminuq
}

define <4 x i64> @test167(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp ugt <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test167:
; AVX512VL: vpmaxuq
}

define <4 x i64> @test168(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp uge <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel

; AVX512VL-LABEL: test168:
; AVX512VL: vpmaxuq
}

define <4 x i64> @test169(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp slt <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test169:
; AVX512VL: vpmaxsq
}

define <4 x i64> @test170(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp sle <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test170:
; AVX512VL: vpmaxsq
}

define <4 x i64> @test171(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp sgt <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test171:
; AVX512VL: vpminsq
}

define <4 x i64> @test172(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp sge <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test172:
; AVX512VL: vpminsq
}

define <4 x i64> @test173(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp ult <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test173:
; AVX512VL: vpmaxuq
}

define <4 x i64> @test174(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp ule <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test174:
; AVX512VL: vpmaxuq
}

define <4 x i64> @test175(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp ugt <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test175:
; AVX512VL: vpminuq
}

define <4 x i64> @test176(<4 x i64> %a, <4 x i64> %b) {
entry:
  %cmp = icmp uge <4 x i64> %a, %b
  %sel = select <4 x i1> %cmp, <4 x i64> %b, <4 x i64> %a
  ret <4 x i64> %sel

; AVX512VL-LABEL: test176:
; AVX512VL: vpminuq
}

define <2 x i64> @test177(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp slt <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test177:
; AVX512VL: vpminsq
}

define <2 x i64> @test178(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp sle <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test178:
; AVX512VL: vpminsq
}

define <2 x i64> @test179(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp sgt <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test179:
; AVX512VL: vpmaxsq
}

define <2 x i64> @test180(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp sge <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test180:
; AVX512VL: vpmaxsq
}

define <2 x i64> @test181(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp ult <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test181:
; AVX512VL: vpminuq
}

define <2 x i64> @test182(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp ule <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test182:
; AVX512VL: vpminuq
}

define <2 x i64> @test183(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp ugt <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test183:
; AVX512VL: vpmaxuq
}

define <2 x i64> @test184(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp uge <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel

; AVX512VL-LABEL: test184:
; AVX512VL: vpmaxuq
}

define <2 x i64> @test185(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp slt <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test185:
; AVX512VL: vpmaxsq
}

define <2 x i64> @test186(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp sle <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test186:
; AVX512VL: vpmaxsq
}

define <2 x i64> @test187(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp sgt <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test187:
; AVX512VL: vpminsq
}

define <2 x i64> @test188(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp sge <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test188:
; AVX512VL: vpminsq
}

define <2 x i64> @test189(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp ult <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test189:
; AVX512VL: vpmaxuq
}

define <2 x i64> @test190(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp ule <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test190:
; AVX512VL: vpmaxuq
}

define <2 x i64> @test191(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp ugt <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test191:
; AVX512VL: vpminuq
}

define <2 x i64> @test192(<2 x i64> %a, <2 x i64> %b) {
entry:
  %cmp = icmp uge <2 x i64> %a, %b
  %sel = select <2 x i1> %cmp, <2 x i64> %b, <2 x i64> %a
  ret <2 x i64> %sel

; AVX512VL-LABEL: test192:
; AVX512VL: vpminuq
}
