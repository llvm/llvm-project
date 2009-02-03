; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep ceq     %t1.s | count 30 
; RUN: grep ceqb    %t1.s | count 10
; RUN: grep ceqhi   %t1.s | count 5
; RUN: grep ceqi    %t1.s | count 5
; RUN: grep cgt     %t1.s | count 30
; RUN: grep cgtb    %t1.s | count 10
; RUN: grep cgthi   %t1.s | count 5
; RUN: grep cgti    %t1.s | count 5
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

declare <4 x i32> @llvm.spu.si.shli(<4 x i32>, i8)

declare <4 x i32> @llvm.spu.si.ceq(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.spu.si.ceqb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.spu.si.ceqh(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.spu.si.ceqi(<4 x i32>, i16)
declare <8 x i16> @llvm.spu.si.ceqhi(<8 x i16>, i16)
declare <16 x i8> @llvm.spu.si.ceqbi(<16 x i8>, i8)

declare <4 x i32> @llvm.spu.si.cgt(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.spu.si.cgtb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.spu.si.cgth(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.spu.si.cgti(<4 x i32>, i16)
declare <8 x i16> @llvm.spu.si.cgthi(<8 x i16>, i16)
declare <16 x i8> @llvm.spu.si.cgtbi(<16 x i8>, i8)

declare <4 x i32> @llvm.spu.si.clgt(<4 x i32>, <4 x i32>)
declare <16 x i8> @llvm.spu.si.clgtb(<16 x i8>, <16 x i8>)
declare <8 x i16> @llvm.spu.si.clgth(<8 x i16>, <8 x i16>)
declare <4 x i32> @llvm.spu.si.clgti(<4 x i32>, i16)
declare <8 x i16> @llvm.spu.si.clgthi(<8 x i16>, i16)
declare <16 x i8> @llvm.spu.si.clgtbi(<16 x i8>, i8)



define <4 x i32> @test(<4 x i32> %A) {
        call <4 x i32> @llvm.spu.si.shli(<4 x i32> %A, i8 3)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <4 x i32> @ceqtest(<4 x i32> %A, <4 x i32> %B) {
        call <4 x i32> @llvm.spu.si.ceq(<4 x i32> %A, <4 x i32> %B)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <8 x i16> @ceqhtest(<8 x i16> %A, <8 x i16> %B) {
        call <8 x i16> @llvm.spu.si.ceqh(<8 x i16> %A, <8 x i16> %B)
        %Y = bitcast <8 x i16> %1 to <8 x i16>
        ret <8 x i16> %Y
}

define <16 x i8> @ceqbtest(<16 x i8> %A, <16 x i8> %B) {
        call <16 x i8> @llvm.spu.si.ceqb(<16 x i8> %A, <16 x i8> %B)
        %Y = bitcast <16 x i8> %1 to <16 x i8>
        ret <16 x i8> %Y
}

define <4 x i32> @ceqitest(<4 x i32> %A) {
        call <4 x i32> @llvm.spu.si.ceqi(<4 x i32> %A, i16 65)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <8 x i16> @ceqhitest(<8 x i16> %A) {
        call <8 x i16> @llvm.spu.si.ceqhi(<8 x i16> %A, i16 65)
        %Y = bitcast <8 x i16> %1 to <8 x i16>
        ret <8 x i16> %Y
}

define <16 x i8> @ceqbitest(<16 x i8> %A) {
        call <16 x i8> @llvm.spu.si.ceqbi(<16 x i8> %A, i8 65)
        %Y = bitcast <16 x i8> %1 to <16 x i8>
        ret <16 x i8> %Y
}

define <4 x i32> @cgttest(<4 x i32> %A, <4 x i32> %B) {
        call <4 x i32> @llvm.spu.si.cgt(<4 x i32> %A, <4 x i32> %B)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <8 x i16> @cgthtest(<8 x i16> %A, <8 x i16> %B) {
        call <8 x i16> @llvm.spu.si.cgth(<8 x i16> %A, <8 x i16> %B)
        %Y = bitcast <8 x i16> %1 to <8 x i16>
        ret <8 x i16> %Y
}

define <16 x i8> @cgtbtest(<16 x i8> %A, <16 x i8> %B) {
        call <16 x i8> @llvm.spu.si.cgtb(<16 x i8> %A, <16 x i8> %B)
        %Y = bitcast <16 x i8> %1 to <16 x i8>
        ret <16 x i8> %Y
}

define <4 x i32> @cgtitest(<4 x i32> %A) {
        call <4 x i32> @llvm.spu.si.cgti(<4 x i32> %A, i16 65)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <8 x i16> @cgthitest(<8 x i16> %A) {
        call <8 x i16> @llvm.spu.si.cgthi(<8 x i16> %A, i16 65)
        %Y = bitcast <8 x i16> %1 to <8 x i16>
        ret <8 x i16> %Y
}

define <16 x i8> @cgtbitest(<16 x i8> %A) {
        call <16 x i8> @llvm.spu.si.cgtbi(<16 x i8> %A, i8 65)
        %Y = bitcast <16 x i8> %1 to <16 x i8>
        ret <16 x i8> %Y
}

define <4 x i32> @clgttest(<4 x i32> %A, <4 x i32> %B) {
        call <4 x i32> @llvm.spu.si.clgt(<4 x i32> %A, <4 x i32> %B)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <8 x i16> @clgthtest(<8 x i16> %A, <8 x i16> %B) {
        call <8 x i16> @llvm.spu.si.clgth(<8 x i16> %A, <8 x i16> %B)
        %Y = bitcast <8 x i16> %1 to <8 x i16>
        ret <8 x i16> %Y
}

define <16 x i8> @clgtbtest(<16 x i8> %A, <16 x i8> %B) {
        call <16 x i8> @llvm.spu.si.clgtb(<16 x i8> %A, <16 x i8> %B)
        %Y = bitcast <16 x i8> %1 to <16 x i8>
        ret <16 x i8> %Y
}

define <4 x i32> @clgtitest(<4 x i32> %A) {
        call <4 x i32> @llvm.spu.si.clgti(<4 x i32> %A, i16 65)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <8 x i16> @clgthitest(<8 x i16> %A) {
        call <8 x i16> @llvm.spu.si.clgthi(<8 x i16> %A, i16 65)
        %Y = bitcast <8 x i16> %1 to <8 x i16>
        ret <8 x i16> %Y
}

define <16 x i8> @clgtbitest(<16 x i8> %A) {
        call <16 x i8> @llvm.spu.si.clgtbi(<16 x i8> %A, i8 65)
        %Y = bitcast <16 x i8> %1 to <16 x i8>
        ret <16 x i8> %Y
}
