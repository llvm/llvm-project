; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep fa      %t1.s | count 5
; RUN: grep fs      %t1.s | count 5
; RUN: grep fm      %t1.s | count 15
; RUN: grep fceq    %t1.s | count 5
; RUN: grep fcmeq   %t1.s | count 5
; RUN: grep fcgt    %t1.s | count 5
; RUN: grep fcmgt   %t1.s | count 5
; RUN: grep fma     %t1.s | count 5
; RUN: grep fnms    %t1.s | count 5
; RUN: grep fms     %t1.s | count 5
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

declare <4 x i32> @llvm.spu.si.shli(<4 x i32>, i8)

declare <4 x float> @llvm.spu.si.fa(<4 x float>, <4 x float>)
declare <4 x float> @llvm.spu.si.fs(<4 x float>, <4 x float>)
declare <4 x float> @llvm.spu.si.fm(<4 x float>, <4 x float>)

declare <4 x float> @llvm.spu.si.fceq(<4 x float>, <4 x float>)
declare <4 x float> @llvm.spu.si.fcmeq(<4 x float>, <4 x float>)
declare <4 x float> @llvm.spu.si.fcgt(<4 x float>, <4 x float>)
declare <4 x float> @llvm.spu.si.fcmgt(<4 x float>, <4 x float>)

declare <4 x float> @llvm.spu.si.fma(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.spu.si.fnms(<4 x float>, <4 x float>, <4 x float>)
declare <4 x float> @llvm.spu.si.fms(<4 x float>, <4 x float>, <4 x float>)

define <4 x i32> @test(<4 x i32> %A) {
        call <4 x i32> @llvm.spu.si.shli(<4 x i32> %A, i8 3)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <4 x float> @fatest(<4 x float> %A, <4 x float> %B) {
        call <4 x float> @llvm.spu.si.fa(<4 x float> %A, <4 x float> %B)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fstest(<4 x float> %A, <4 x float> %B) {
        call <4 x float> @llvm.spu.si.fs(<4 x float> %A, <4 x float> %B)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fmtest(<4 x float> %A, <4 x float> %B) {
        call <4 x float> @llvm.spu.si.fm(<4 x float> %A, <4 x float> %B)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fceqtest(<4 x float> %A, <4 x float> %B) {
        call <4 x float> @llvm.spu.si.fceq(<4 x float> %A, <4 x float> %B)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fcmeqtest(<4 x float> %A, <4 x float> %B) {
        call <4 x float> @llvm.spu.si.fcmeq(<4 x float> %A, <4 x float> %B)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fcgttest(<4 x float> %A, <4 x float> %B) {
        call <4 x float> @llvm.spu.si.fcgt(<4 x float> %A, <4 x float> %B)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fcmgttest(<4 x float> %A, <4 x float> %B) {
        call <4 x float> @llvm.spu.si.fcmgt(<4 x float> %A, <4 x float> %B)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fmatest(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
        call <4 x float> @llvm.spu.si.fma(<4 x float> %A, <4 x float> %B, <4 x float> %C)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fnmstest(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
        call <4 x float> @llvm.spu.si.fnms(<4 x float> %A, <4 x float> %B, <4 x float> %C)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}

define <4 x float> @fmstest(<4 x float> %A, <4 x float> %B, <4 x float> %C) {
        call <4 x float> @llvm.spu.si.fms(<4 x float> %A, <4 x float> %B, <4 x float> %C)
        %Y = bitcast <4 x float> %1 to <4 x float>
        ret <4 x float> %Y
}
