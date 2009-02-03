; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep and       %t1.s | count 20
; RUN: grep andc      %t1.s | count 5
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

declare <4 x i32> @llvm.spu.si.and(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.spu.si.andc(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.spu.si.andi(<4 x i32>, i16)
declare <8 x i16> @llvm.spu.si.andhi(<8 x i16>, i16)
declare <16 x i8> @llvm.spu.si.andbi(<16 x i8>, i8)

declare <4 x i32> @llvm.spu.si.or(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.spu.si.orc(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.spu.si.ori(<4 x i32>, i16)
declare <8 x i16> @llvm.spu.si.orhi(<8 x i16>, i16)
declare <16 x i8> @llvm.spu.si.orbi(<16 x i8>, i8)

declare <4 x i32> @llvm.spu.si.xor(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.spu.si.xori(<4 x i32>, i16)
declare <8 x i16> @llvm.spu.si.xorhi(<8 x i16>, i16)
declare <16 x i8> @llvm.spu.si.xorbi(<16 x i8>, i8)

declare <4 x i32> @llvm.spu.si.nand(<4 x i32>, <4 x i32>)
declare <4 x i32> @llvm.spu.si.nor(<4 x i32>, <4 x i32>)

define <4 x i32> @andtest(<4 x i32> %A, <4 x i32> %B) {
        call <4 x i32> @llvm.spu.si.and(<4 x i32> %A, <4 x i32> %B)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <4 x i32> @andctest(<4 x i32> %A, <4 x i32> %B) {
        call <4 x i32> @llvm.spu.si.andc(<4 x i32> %A, <4 x i32> %B)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <4 x i32> @anditest(<4 x i32> %A) {
        call <4 x i32> @llvm.spu.si.andi(<4 x i32> %A, i16 65)
        %Y = bitcast <4 x i32> %1 to <4 x i32>
        ret <4 x i32> %Y
}

define <8 x i16> @andhitest(<8 x i16> %A) {
        call <8 x i16> @llvm.spu.si.andhi(<8 x i16> %A, i16 65)
        %Y = bitcast <8 x i16> %1 to <8 x i16>
        ret <8 x i16> %Y
}
