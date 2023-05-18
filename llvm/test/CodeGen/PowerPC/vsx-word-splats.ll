; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-BE

define <4 x float> @test0f(<4 x float> %a) {
entry:
  %0 = bitcast <4 x float> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  %2 = bitcast <16 x i8> %1 to <4 x float>
  ret <4 x float> %2
; CHECK-LABEL: test0f
; CHECK: xxspltw 34, 34, 3
; CHECK-BE-LABEL: test0f
; CHECK-BE: xxspltw 34, 34, 0
}

define <4 x float> @test1f(<4 x float> %a) {
entry:
  %0 = bitcast <4 x float> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7>
  %2 = bitcast <16 x i8> %1 to <4 x float>
  ret <4 x float> %2
; CHECK-LABEL: test1f
; CHECK: xxspltw 34, 34, 2
; CHECK-BE-LABEL: test1f
; CHECK-BE: xxspltw 34, 34, 1
}

define <4 x float> @test2f(<4 x float> %a) {
entry:
  %0 = bitcast <4 x float> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11>
  %2 = bitcast <16 x i8> %1 to <4 x float>
  ret <4 x float> %2
; CHECK-LABEL: test2f
; CHECK: xxspltw 34, 34, 1
; CHECK-BE-LABEL: test2f
; CHECK-BE: xxspltw 34, 34, 2
}

define <4 x float> @test3f(<4 x float> %a) {
entry:
  %0 = bitcast <4 x float> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15>
  %2 = bitcast <16 x i8> %1 to <4 x float>
  ret <4 x float> %2
; CHECK-LABEL: test3f
; CHECK: xxspltw 34, 34, 0
; CHECK-BE-LABEL: test3f
; CHECK-BE: xxspltw 34, 34, 3
}

define <4 x i32> @test0si(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test0si
; CHECK: xxspltw 34, 34, 3
; CHECK-BE-LABEL: test0si
; CHECK-BE: xxspltw 34, 34, 0
}

define <4 x i32> @test1si(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test1si
; CHECK: xxspltw 34, 34, 2
; CHECK-BE-LABEL: test1si
; CHECK-BE: xxspltw 34, 34, 1
}

define <4 x i32> @test2si(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test2si
; CHECK: xxspltw 34, 34, 1
; CHECK-BE-LABEL: test2si
; CHECK-BE: xxspltw 34, 34, 2
}

define <4 x i32> @test3si(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test3si
; CHECK: xxspltw 34, 34, 0
; CHECK-BE-LABEL: test3si
; CHECK-BE: xxspltw 34, 34, 3
}

define <4 x i32> @test0ui(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test0ui
; CHECK: xxspltw 34, 34, 3
; CHECK-BE-LABEL: test0ui
; CHECK-BE: xxspltw 34, 34, 0
}

define <4 x i32> @test1ui(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7, i32 4, i32 5, i32 6, i32 7>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test1ui
; CHECK: xxspltw 34, 34, 2
; CHECK-BE-LABEL: test1ui
; CHECK-BE: xxspltw 34, 34, 1
}

define <4 x i32> @test2ui(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11, i32 8, i32 9, i32 10, i32 11>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test2ui
; CHECK: xxspltw 34, 34, 1
; CHECK-BE-LABEL: test2ui
; CHECK-BE: xxspltw 34, 34, 2
}

define <4 x i32> @test3ui(<4 x i32> %a) {
entry:
  %0 = bitcast <4 x i32> %a to <16 x i8>
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> <i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15, i32 12, i32 13, i32 14, i32 15>
  %2 = bitcast <16 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
; CHECK-LABEL: test3ui
; CHECK: xxspltw 34, 34, 0
; CHECK-BE-LABEL: test3ui
; CHECK-BE: xxspltw 34, 34, 3
}
