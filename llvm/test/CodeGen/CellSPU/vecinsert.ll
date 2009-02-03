; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep cbd     %t1.s | count 5
; RUN: grep chd     %t1.s | count 5
; RUN: grep cwd     %t1.s | count 10
; RUN: grep -w il   %t1.s | count 5
; RUN: grep -w ilh  %t1.s | count 6
; RUN: grep iohl    %t1.s | count 1
; RUN: grep ilhu    %t1.s | count 4
; RUN: grep shufb   %t1.s | count 26
; RUN: grep 17219   %t1.s | count 1 
; RUN: grep 22598   %t1.s | count 1
; RUN: grep -- -39  %t1.s | count 1
; RUN: grep    24   %t1.s | count 1
; RUN: grep  1159   %t1.s | count 1
; ModuleID = 'vecinsert.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128"
target triple = "spu-unknown-elf"

; 67 -> 0x43, as 8-bit vector constant load = 0x4343 (17219)0x4343
define <16 x i8> @test_v16i8(<16 x i8> %P, i8 %x) {
entry:
        %tmp1 = insertelement <16 x i8> %P, i8 %x, i32 10
        %tmp1.1 = insertelement <16 x i8> %tmp1, i8 67, i32 7
        %tmp1.2 = insertelement <16 x i8> %tmp1.1, i8 %x, i32 15
        ret <16 x i8> %tmp1.2
}

; 22598 -> 0x5846
define <8 x i16> @test_v8i16(<8 x i16> %P, i16 %x) {
entry:
        %tmp1 = insertelement <8 x i16> %P, i16 %x, i32 5
        %tmp1.1 = insertelement <8 x i16> %tmp1, i16 22598, i32 7
        %tmp1.2 = insertelement <8 x i16> %tmp1.1, i16 %x, i32 2
        ret <8 x i16> %tmp1.2
}

; 1574023 -> 0x180487 (ILHU 24/IOHL 1159)
define <4 x i32> @test_v4i32_1(<4 x i32> %P, i32 %x) {
entry:
        %tmp1 = insertelement <4 x i32> %P, i32 %x, i32 2
        %tmp1.1 = insertelement <4 x i32> %tmp1, i32 1574023, i32 1
        %tmp1.2 = insertelement <4 x i32> %tmp1.1, i32 %x, i32 3
        ret <4 x i32> %tmp1.2
}

; Should generate IL for the load
define <4 x i32> @test_v4i32_2(<4 x i32> %P, i32 %x) {
entry:
        %tmp1 = insertelement <4 x i32> %P, i32 %x, i32 2
        %tmp1.1 = insertelement <4 x i32> %tmp1, i32 -39, i32 1
        %tmp1.2 = insertelement <4 x i32> %tmp1.1, i32 %x, i32 3
        ret <4 x i32> %tmp1.2
}

define void @variable_v16i8_1(<16 x i8>* %a, i32 %i) nounwind {
entry:
	%arrayidx = getelementptr <16 x i8>* %a, i32 %i
	%tmp2 = load <16 x i8>* %arrayidx
	%tmp3 = insertelement <16 x i8> %tmp2, i8 1, i32 1
	%tmp8 = insertelement <16 x i8> %tmp3, i8 2, i32 11
	store <16 x i8> %tmp8, <16 x i8>* %arrayidx
	ret void
}

define void @variable_v8i16_1(<8 x i16>* %a, i32 %i) nounwind {
entry:
	%arrayidx = getelementptr <8 x i16>* %a, i32 %i
	%tmp2 = load <8 x i16>* %arrayidx
	%tmp3 = insertelement <8 x i16> %tmp2, i16 1, i32 1
	%tmp8 = insertelement <8 x i16> %tmp3, i16 2, i32 6
	store <8 x i16> %tmp8, <8 x i16>* %arrayidx
	ret void
}

define void @variable_v4i32_1(<4 x i32>* %a, i32 %i) nounwind {
entry:
	%arrayidx = getelementptr <4 x i32>* %a, i32 %i
	%tmp2 = load <4 x i32>* %arrayidx
	%tmp3 = insertelement <4 x i32> %tmp2, i32 1, i32 1
	%tmp8 = insertelement <4 x i32> %tmp3, i32 2, i32 2
	store <4 x i32> %tmp8, <4 x i32>* %arrayidx
	ret void
}

define void @variable_v4f32_1(<4 x float>* %a, i32 %i) nounwind {
entry:
	%arrayidx = getelementptr <4 x float>* %a, i32 %i
	%tmp2 = load <4 x float>* %arrayidx
	%tmp3 = insertelement <4 x float> %tmp2, float 1.000000e+00, i32 1
	%tmp8 = insertelement <4 x float> %tmp3, float 2.000000e+00, i32 2
	store <4 x float> %tmp8, <4 x float>* %arrayidx
	ret void
}

define void @variable_v2i64_1(<2 x i64>* %a, i32 %i) nounwind {
entry:
	%arrayidx = getelementptr <2 x i64>* %a, i32 %i
	%tmp2 = load <2 x i64>* %arrayidx
	%tmp3 = insertelement <2 x i64> %tmp2, i64 615, i32 0
	store <2 x i64> %tmp3, <2 x i64>* %arrayidx
	ret void
}

define void @variable_v2i64_2(<2 x i64>* %a, i32 %i) nounwind {
entry:
	%arrayidx = getelementptr <2 x i64>* %a, i32 %i
	%tmp2 = load <2 x i64>* %arrayidx
	%tmp3 = insertelement <2 x i64> %tmp2, i64 615, i32 1
	store <2 x i64> %tmp3, <2 x i64>* %arrayidx
	ret void
}

define void @variable_v2f64_1(<2 x double>* %a, i32 %i) nounwind {
entry:
	%arrayidx = getelementptr <2 x double>* %a, i32 %i
	%tmp2 = load <2 x double>* %arrayidx
	%tmp3 = insertelement <2 x double> %tmp2, double 1.000000e+00, i32 1
	store <2 x double> %tmp3, <2 x double>* %arrayidx
	ret void
}
