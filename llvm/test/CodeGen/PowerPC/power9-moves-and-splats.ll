; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-BE

@Globi = external global i32, align 4
@Globf = external global float, align 4

define <2 x i64> @test1(i64 %a, i64 %b) {
entry:
; CHECK-LABEL: test1
; CHECK: mtvsrdd 34, 4, 3
; CHECK-BE-LABEL: test1
; CHECK-BE: mtvsrdd 34, 3, 4
  %vecins = insertelement <2 x i64> undef, i64 %a, i32 0
  %vecins1 = insertelement <2 x i64> %vecins, i64 %b, i32 1
  ret <2 x i64> %vecins1
}

define i64 @test2(<2 x i64> %a) {
entry:
; CHECK-LABEL: test2
; CHECK: mfvsrld 3, 34
  %0 = extractelement <2 x i64> %a, i32 0
  ret i64 %0
}

define i64 @test3(<2 x i64> %a) {
entry:
; CHECK-BE-LABEL: test3
; CHECK-BE: mfvsrld 3, 34
  %0 = extractelement <2 x i64> %a, i32 1
  ret i64 %0
}

define <4 x i32> @test4(i32* nocapture readonly %in) {
entry:
; CHECK-LABEL: test4
; CHECK: lxvwsx 34, 0, 3
; CHECK-NOT: xxspltw
; CHECK-BE-LABEL: test4
; CHECK-BE: lxvwsx 34, 0, 3
; CHECK-BE-NOT: xxspltw
  %0 = load i32, i32* %in, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
}

define <4 x float> @test5(float* nocapture readonly %in) {
entry:
; CHECK-LABEL: test5
; CHECK: lxvwsx 34, 0, 3
; CHECK-NOT: xxspltw
; CHECK-BE-LABEL: test5
; CHECK-BE: lxvwsx 34, 0, 3
; CHECK-BE-NOT: xxspltw
  %0 = load float, float* %in, align 4
  %splat.splatinsert = insertelement <4 x float> undef, float %0, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
}

define <4 x i32> @test6() {
entry:
; CHECK-LABEL: test6
; CHECK: addis
; CHECK: ld [[TOC:[0-9]+]], .LC0
; CHECK: lxvwsx 34, 0, 3
; CHECK-NOT: xxspltw
; CHECK-BE-LABEL: test6
; CHECK-BE: addis
; CHECK-BE: ld [[TOC:[0-9]+]], .LC0
; CHECK-BE: lxvwsx 34, 0, 3
; CHECK-BE-NOT: xxspltw
  %0 = load i32, i32* @Globi, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
}

define <4 x float> @test7() {
entry:
; CHECK-LABEL: test7
; CHECK: addis
; CHECK: ld [[TOC:[0-9]+]], .LC1
; CHECK: lxvwsx 34, 0, 3
; CHECK-NOT: xxspltw
; CHECK-BE-LABEL: test7
; CHECK-BE: addis
; CHECK-BE: ld [[TOC:[0-9]+]], .LC1
; CHECK-BE: lxvwsx 34, 0, 3
; CHECK-BE-NOT: xxspltw
  %0 = load float, float* @Globf, align 4
  %splat.splatinsert = insertelement <4 x float> undef, float %0, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
}

define <16 x i8> @test8() {
entry:
; CHECK-LABEL: test8
; CHECK: xxlxor 34, 34, 34
; CHECK-BE-LABEL: test8
; CHECK-BE: xxlxor 34, 34, 34
  ret <16 x i8> zeroinitializer
}

define <16 x i8> @test9() {
entry:
; CHECK-LABEL: test9
; CHECK: xxspltib 34, 1
; CHECK-BE-LABEL: test9
; CHECK-BE: xxspltib 34, 1
  ret <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>
}

define <16 x i8> @test10() {
entry:
; CHECK-LABEL: test10
; CHECK: xxspltib 34, 127
; CHECK-BE-LABEL: test10
; CHECK-BE: xxspltib 34, 127
  ret <16 x i8> <i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127>
}

define <16 x i8> @test11() {
entry:
; CHECK-LABEL: test11
; CHECK: xxspltib 34, 128
; CHECK-BE-LABEL: test11
; CHECK-BE: xxspltib 34, 128
  ret <16 x i8> <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
}

define <16 x i8> @test12() {
entry:
; CHECK-LABEL: test12
; CHECK: xxspltib 34, 255
; CHECK-BE-LABEL: test12
; CHECK-BE: xxspltib 34, 255
  ret <16 x i8> <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
}

define <16 x i8> @test13() {
entry:
; CHECK-LABEL: test13
; CHECK: xxspltib 34, 129
; CHECK-BE-LABEL: test13
; CHECK-BE: xxspltib 34, 129
  ret <16 x i8> <i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127, i8 -127>
}

define <4 x i32> @test14(<4 x i32> %a, i32* nocapture readonly %b) {
entry:
; CHECK-LABEL: test14
; CHECK: lwz [[LD:[0-9]+]],
; CHECK: mtvsrws 34, [[LD]]
; CHECK-BE-LABEL: test14
; CHECK-BE: lwz [[LD:[0-9]+]],
; CHECK-BE: mtvsrws 34, [[LD]]
  %0 = load i32, i32* %b, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  %1 = add i32 %0, 5
  store i32 %1, i32* %b, align 4
  ret <4 x i32> %splat.splat
}
