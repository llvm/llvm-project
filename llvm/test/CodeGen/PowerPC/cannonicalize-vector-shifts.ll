; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

; RUN: llc -mcpu=pwr8 -vec-extabi -mtriple=powerpc64-ibm-aix-xcoff \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr8 -vec-extabi -mtriple=powerpc-ibm-aix-xcoff \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

define <4 x i32> @test1(<4 x i32> %a) {
entry:
; CHECK-LABEL: test1
; CHECK: xxswapd 34, 34
  %vecins6 = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
  ret <4 x i32> %vecins6
}

define <8 x i16> @test2(<8 x i16> %a) #0 {
entry:
; CHECK-LABEL: test2
; CHECK: xxswapd 34, 34
  %vecins14 = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <8 x i16> %vecins14
}

define <16 x i8> @test3(<16 x i8> %a) #0 {
entry:
; CHECK-LABEL: test3
; CHECK: xxswapd 34, 34
  %vecins30 = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <16 x i8> %vecins30
}
