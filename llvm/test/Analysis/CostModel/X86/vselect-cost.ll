; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2,-sse4.1 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=SSE41
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7-avx -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2 -cost-model -analyze | FileCheck %s -check-prefix=CHECK -check-prefix=AVX2


; Verify the cost of vector select instructions.

; SSE41 added blend instructions with an immediate for <2 x double> and
; <4 x float>. Integers of the same size should also use those instructions.

define <2 x i64> @test_2i64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_2i64':
; SSE2: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
; SSE41: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
  %sel = select <2 x i1> <i1 true, i1 false>, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %sel
}

define <2 x double> @test_2double(<2 x double> %a, <2 x double> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_2double':
; SSE2: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
; SSE41: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <2 x i1>
  %sel = select <2 x i1> <i1 true, i1 false>, <2 x double> %a, <2 x double> %b
  ret <2 x double> %sel
}

define <4 x i32> @test_4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_4i32':
; SSE2: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; SSE41: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
  %sel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x i32> %a, <4 x i32> %b
  ret <4 x i32> %sel
}

define <4 x float> @test_4float(<4 x float> %a, <4 x float> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_4float':
; SSE2: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; SSE41: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
  %sel = select <4 x i1> <i1 true, i1 false, i1 true, i1 true>, <4 x float> %a, <4 x float> %b
  ret <4 x float> %sel
}

define <16 x i8> @test_16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_16i8':
; SSE2: Cost Model: {{.*}} 1 for instruction:   %sel = select <16 x i1>
; SSE41: Cost Model: {{.*}} 1 for instruction:   %sel = select <16 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <16 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <16 x i1>
  %sel = select <16 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true>, <16 x i8> %a, <16 x i8> %b
  ret <16 x i8> %sel
}

; AVX added blend instructions with an immediate for <4 x double> and
; <8 x float>. Integers of the same size should also use those instructions.
define <4 x i64> @test_4i64(<4 x i64> %a, <4 x i64> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_4i64':
; SSE2: Cost Model: {{.*}} 2 for instruction:   %sel = select <4 x i1>
; SSE41: Cost Model: {{.*}} 2 for instruction:   %sel = select <4 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
  %sel = select <4 x i1> <i1 true, i1 false, i1 false, i1 true>, <4 x i64> %a, <4 x i64> %b
  ret <4 x i64> %sel
}

define <4 x double> @test_4double(<4 x double> %a, <4 x double> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_4double':
; SSE2: Cost Model: {{.*}} 2 for instruction:   %sel = select <4 x i1>
; SSE41: Cost Model: {{.*}} 2 for instruction:   %sel = select <4 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <4 x i1>
  %sel = select <4 x i1> <i1 true, i1 false, i1 true, i1 false>, <4 x double> %a, <4 x double> %b
  ret <4 x double> %sel
}

define <8 x i32> @test_8i32(<8 x i32> %a, <8 x i32> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_8i32':
; SSE2: Cost Model: {{.*}} 2 for instruction:   %sel = select <8 x i1>
; SSE41: Cost Model: {{.*}} 2 for instruction:   %sel = select <8 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <8 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <8 x i1>
  %sel = select <8 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 false>, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %sel
}

define <8 x float> @test_8float(<8 x float> %a, <8 x float> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_8float':
; SSE2: Cost Model: {{.*}} 2 for instruction:   %sel = select <8 x i1>
; SSE41: Cost Model: {{.*}} 2 for instruction:   %sel = select <8 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <8 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <8 x i1>
  %sel = select <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <8 x float> %a, <8 x float> %b
  ret <8 x float> %sel
}

; AVX2
define <16 x i16> @test_16i16(<16 x i16> %a, <16 x i16> %b) {
; CHECK:Printing analysis 'Cost Model Analysis' for function 'test_16i16':
; SSE2: Cost Model: {{.*}} 2 for instruction:   %sel = select <16 x i1>
; SSE41: Cost Model: {{.*}} 2 for instruction:   %sel = select <16 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <16 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <16 x i1>
  %sel = select <16 x i1> <i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false>, <16 x i16> %a, <16 x i16> %b
  ret <16 x i16> %sel
}

define <32 x i8> @test_32i8(<32 x i8> %a, <32 x i8> %b) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'test_32i8':
; SSE2: Cost Model: {{.*}} 2 for instruction:   %sel = select <32 x i1>
; SSE41: Cost Model: {{.*}} 2 for instruction:   %sel = select <32 x i1>
; AVX: Cost Model: {{.*}} 1 for instruction:   %sel = select <32 x i1>
; AVX2: Cost Model: {{.*}} 1 for instruction:   %sel = select <32 x i1>
  %sel = select <32 x i1> <i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true>, <32 x i8> %a, <32 x i8> %b
  ret <32 x i8> %sel
}

