; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep {stqd.*0(\$3)}      %t1.s | count 4
; RUN: grep {stqd.*16(\$3)}     %t1.s | count 4
; RUN: grep 16256               %t1.s | count 2
; RUN: grep 16384               %t1.s | count 1
; RUN: grep 771                 %t1.s | count 4
; RUN: grep 515                 %t1.s | count 2
; RUN: grep 1799                %t1.s | count 2
; RUN: grep 1543                %t1.s | count 5
; RUN: grep 1029                %t1.s | count 3
; RUN: grep {shli.*, 4}         %t1.s | count 4
; RUN: grep stqx                %t1.s | count 4
; RUN: grep ilhu                %t1.s | count 11
; RUN: grep iohl                %t1.s | count 8
; RUN: grep shufb               %t1.s | count 15
; RUN: grep frds                %t1.s | count 1

; ModuleID = 'stores.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define void @store_v16i8_1(<16 x i8>* %a) nounwind {
entry:
	store <16 x i8> < i8 1, i8 2, i8 1, i8 1, i8 1, i8 2, i8 1, i8 1, i8 1, i8 2, i8 1, i8 1, i8 1, i8 2, i8 1, i8 1 >, <16 x i8>* %a
	ret void
}

define void @store_v16i8_2(<16 x i8>* %a) nounwind {
entry:
	%arrayidx = getelementptr <16 x i8>* %a, i32 1
	store <16 x i8> < i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2 >, <16 x i8>* %arrayidx
	ret void
}

define void @store_v16i8_3(<16 x i8>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <16 x i8>* %a, i32 %i
	store <16 x i8> < i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1 >, <16 x i8>* %arrayidx
        ret void
}

define void @store_v8i16_1(<8 x i16>* %a) nounwind {
entry:
	store <8 x i16> < i16 1, i16 2, i16 1, i16 1, i16 1, i16 2, i16 1, i16 1 >, <8 x i16>* %a
	ret void
}

define void @store_v8i16_2(<8 x i16>* %a) nounwind {
entry:
	%arrayidx = getelementptr <8 x i16>* %a, i16 1
	store <8 x i16> < i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2 >, <8 x i16>* %arrayidx
	ret void
}

define void @store_v8i16_3(<8 x i16>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <8 x i16>* %a, i32 %i
	store <8 x i16> < i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1 >, <8 x i16>* %arrayidx
        ret void
}

define void @store_v4i32_1(<4 x i32>* %a) nounwind {
entry:
	store <4 x i32> < i32 1, i32 2, i32 1, i32 1 >, <4 x i32>* %a
	ret void
}

define void @store_v4i32_2(<4 x i32>* %a) nounwind {
entry:
	%arrayidx = getelementptr <4 x i32>* %a, i32 1
	store <4 x i32> < i32 2, i32 2, i32 2, i32 2 >, <4 x i32>* %arrayidx
	ret void
}

define void @store_v4i32_3(<4 x i32>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <4 x i32>* %a, i32 %i
        store <4 x i32> < i32 1, i32 1, i32 1, i32 1 >, <4 x i32>* %arrayidx
        ret void
}

define void @store_v4f32_1(<4 x float>* %a) nounwind {
entry:
	store <4 x float> < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >, <4 x float>* %a
	ret void
}

define void @store_v4f32_2(<4 x float>* %a) nounwind {
entry:
	%arrayidx = getelementptr <4 x float>* %a, i32 1
	store <4 x float> < float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00 >, <4 x float>* %arrayidx
	ret void
}

define void @store_v4f32_3(<4 x float>* %a, i32 %i) nounwind {
entry:
        %arrayidx = getelementptr <4 x float>* %a, i32 %i
        store <4 x float> < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >, <4 x float>* %arrayidx
        ret void
}

; Test truncating stores:

define zeroext i8 @tstore_i16_i8(i16 signext %val, i8* %dest) nounwind {
entry:
	%conv = trunc i16 %val to i8
	store i8 %conv, i8* %dest
	ret i8 %conv
}

define zeroext i8 @tstore_i32_i8(i32 %val, i8* %dest) nounwind {
entry:
	%conv = trunc i32 %val to i8
	store i8 %conv, i8* %dest
	ret i8 %conv
}

define signext i16 @tstore_i32_i16(i32 %val, i16* %dest) nounwind {
entry:
	%conv = trunc i32 %val to i16
	store i16 %conv, i16* %dest
	ret i16 %conv
}

define zeroext i8 @tstore_i64_i8(i64 %val, i8* %dest) nounwind {
entry:
	%conv = trunc i64 %val to i8
	store i8 %conv, i8* %dest
	ret i8 %conv
}

define signext i16 @tstore_i64_i16(i64 %val, i16* %dest) nounwind {
entry:
	%conv = trunc i64 %val to i16
	store i16 %conv, i16* %dest
	ret i16 %conv
}

define i32 @tstore_i64_i32(i64 %val, i32* %dest) nounwind {
entry:
	%conv = trunc i64 %val to i32
	store i32 %conv, i32* %dest
	ret i32 %conv
}

define float @tstore_f64_f32(double %val, float* %dest) nounwind {
entry:
	%conv = fptrunc double %val to float
	store float %conv, float* %dest
	ret float %conv
}
