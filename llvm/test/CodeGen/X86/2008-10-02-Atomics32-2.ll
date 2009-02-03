; RUN: llvm-as < %s | llc -march=x86
;; This version includes 64-bit version of binary operators (in 32-bit mode).
;; Swap, cmp-and-swap not supported yet in this mode.
; ModuleID = 'Atomics.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@sc = common global i8 0		; <i8*> [#uses=52]
@uc = common global i8 0		; <i8*> [#uses=112]
@ss = common global i16 0		; <i16*> [#uses=15]
@us = common global i16 0		; <i16*> [#uses=15]
@si = common global i32 0		; <i32*> [#uses=15]
@ui = common global i32 0		; <i32*> [#uses=23]
@sl = common global i32 0		; <i32*> [#uses=15]
@ul = common global i32 0		; <i32*> [#uses=15]
@sll = common global i64 0, align 8		; <i64*> [#uses=13]
@ull = common global i64 0, align 8		; <i64*> [#uses=13]

define void @test_op_ignore() nounwind {
entry:
	%0 = call i8 @llvm.atomic.load.add.i8.p0i8(i8* @sc, i8 1)		; <i8> [#uses=0]
	%1 = call i8 @llvm.atomic.load.add.i8.p0i8(i8* @uc, i8 1)		; <i8> [#uses=0]
	%2 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%3 = call i16 @llvm.atomic.load.add.i16.p0i16(i16* %2, i16 1)		; <i16> [#uses=0]
	%4 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%5 = call i16 @llvm.atomic.load.add.i16.p0i16(i16* %4, i16 1)		; <i16> [#uses=0]
	%6 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%7 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %6, i32 1)		; <i32> [#uses=0]
	%8 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%9 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %8, i32 1)		; <i32> [#uses=0]
	%10 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%11 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %10, i32 1)		; <i32> [#uses=0]
	%12 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%13 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %12, i32 1)		; <i32> [#uses=0]
	%14 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%15 = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %14, i64 1)		; <i64> [#uses=0]
	%16 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%17 = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %16, i64 1)		; <i64> [#uses=0]
	%18 = call i8 @llvm.atomic.load.sub.i8.p0i8(i8* @sc, i8 1)		; <i8> [#uses=0]
	%19 = call i8 @llvm.atomic.load.sub.i8.p0i8(i8* @uc, i8 1)		; <i8> [#uses=0]
	%20 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%21 = call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %20, i16 1)		; <i16> [#uses=0]
	%22 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%23 = call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %22, i16 1)		; <i16> [#uses=0]
	%24 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%25 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %24, i32 1)		; <i32> [#uses=0]
	%26 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%27 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %26, i32 1)		; <i32> [#uses=0]
	%28 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%29 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %28, i32 1)		; <i32> [#uses=0]
	%30 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%31 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %30, i32 1)		; <i32> [#uses=0]
	%32 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%33 = call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %32, i64 1)		; <i64> [#uses=0]
	%34 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%35 = call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %34, i64 1)		; <i64> [#uses=0]
	%36 = call i8 @llvm.atomic.load.or.i8.p0i8(i8* @sc, i8 1)		; <i8> [#uses=0]
	%37 = call i8 @llvm.atomic.load.or.i8.p0i8(i8* @uc, i8 1)		; <i8> [#uses=0]
	%38 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%39 = call i16 @llvm.atomic.load.or.i16.p0i16(i16* %38, i16 1)		; <i16> [#uses=0]
	%40 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%41 = call i16 @llvm.atomic.load.or.i16.p0i16(i16* %40, i16 1)		; <i16> [#uses=0]
	%42 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%43 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %42, i32 1)		; <i32> [#uses=0]
	%44 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%45 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %44, i32 1)		; <i32> [#uses=0]
	%46 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%47 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %46, i32 1)		; <i32> [#uses=0]
	%48 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%49 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %48, i32 1)		; <i32> [#uses=0]
	%50 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%51 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %50, i64 1)		; <i64> [#uses=0]
	%52 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%53 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %52, i64 1)		; <i64> [#uses=0]
	%54 = call i8 @llvm.atomic.load.xor.i8.p0i8(i8* @sc, i8 1)		; <i8> [#uses=0]
	%55 = call i8 @llvm.atomic.load.xor.i8.p0i8(i8* @uc, i8 1)		; <i8> [#uses=0]
	%56 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%57 = call i16 @llvm.atomic.load.xor.i16.p0i16(i16* %56, i16 1)		; <i16> [#uses=0]
	%58 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%59 = call i16 @llvm.atomic.load.xor.i16.p0i16(i16* %58, i16 1)		; <i16> [#uses=0]
	%60 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%61 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %60, i32 1)		; <i32> [#uses=0]
	%62 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%63 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %62, i32 1)		; <i32> [#uses=0]
	%64 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%65 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %64, i32 1)		; <i32> [#uses=0]
	%66 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%67 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %66, i32 1)		; <i32> [#uses=0]
	%68 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%69 = call i64 @llvm.atomic.load.xor.i64.p0i64(i64* %68, i64 1)		; <i64> [#uses=0]
	%70 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%71 = call i64 @llvm.atomic.load.xor.i64.p0i64(i64* %70, i64 1)		; <i64> [#uses=0]
	%72 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @sc, i8 1)		; <i8> [#uses=0]
	%73 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @uc, i8 1)		; <i8> [#uses=0]
	%74 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%75 = call i16 @llvm.atomic.load.and.i16.p0i16(i16* %74, i16 1)		; <i16> [#uses=0]
	%76 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%77 = call i16 @llvm.atomic.load.and.i16.p0i16(i16* %76, i16 1)		; <i16> [#uses=0]
	%78 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%79 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %78, i32 1)		; <i32> [#uses=0]
	%80 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%81 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %80, i32 1)		; <i32> [#uses=0]
	%82 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%83 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %82, i32 1)		; <i32> [#uses=0]
	%84 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%85 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %84, i32 1)		; <i32> [#uses=0]
	%86 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%87 = call i64 @llvm.atomic.load.and.i64.p0i64(i64* %86, i64 1)		; <i64> [#uses=0]
	%88 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%89 = call i64 @llvm.atomic.load.and.i64.p0i64(i64* %88, i64 1)		; <i64> [#uses=0]
	%90 = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* @sc, i8 1)		; <i8> [#uses=0]
	%91 = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* @uc, i8 1)		; <i8> [#uses=0]
	%92 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%93 = call i16 @llvm.atomic.load.nand.i16.p0i16(i16* %92, i16 1)		; <i16> [#uses=0]
	%94 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%95 = call i16 @llvm.atomic.load.nand.i16.p0i16(i16* %94, i16 1)		; <i16> [#uses=0]
	%96 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%97 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %96, i32 1)		; <i32> [#uses=0]
	%98 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%99 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %98, i32 1)		; <i32> [#uses=0]
	%100 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%101 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %100, i32 1)		; <i32> [#uses=0]
	%102 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%103 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %102, i32 1)		; <i32> [#uses=0]
	%104 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%105 = call i64 @llvm.atomic.load.nand.i64.p0i64(i64* %104, i64 1)		; <i64> [#uses=0]
	%106 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%107 = call i64 @llvm.atomic.load.nand.i64.p0i64(i64* %106, i64 1)		; <i64> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.load.add.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.add.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.add.i32.p0i32(i32*, i32) nounwind

declare i64 @llvm.atomic.load.add.i64.p0i64(i64*, i64) nounwind

declare i8 @llvm.atomic.load.sub.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.sub.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.sub.i32.p0i32(i32*, i32) nounwind

declare i64 @llvm.atomic.load.sub.i64.p0i64(i64*, i64) nounwind

declare i8 @llvm.atomic.load.or.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.or.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.or.i32.p0i32(i32*, i32) nounwind

declare i64 @llvm.atomic.load.or.i64.p0i64(i64*, i64) nounwind

declare i8 @llvm.atomic.load.xor.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.xor.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.xor.i32.p0i32(i32*, i32) nounwind

declare i64 @llvm.atomic.load.xor.i64.p0i64(i64*, i64) nounwind

declare i8 @llvm.atomic.load.and.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.and.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.and.i32.p0i32(i32*, i32) nounwind

declare i64 @llvm.atomic.load.and.i64.p0i64(i64*, i64) nounwind

declare i8 @llvm.atomic.load.nand.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.nand.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.nand.i32.p0i32(i32*, i32) nounwind

declare i64 @llvm.atomic.load.nand.i64.p0i64(i64*, i64) nounwind

define void @test_fetch_and_op() nounwind {
entry:
	%0 = call i8 @llvm.atomic.load.add.i8.p0i8(i8* @sc, i8 11)		; <i8> [#uses=1]
	store i8 %0, i8* @sc, align 1
	%1 = call i8 @llvm.atomic.load.add.i8.p0i8(i8* @uc, i8 11)		; <i8> [#uses=1]
	store i8 %1, i8* @uc, align 1
	%2 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%3 = call i16 @llvm.atomic.load.add.i16.p0i16(i16* %2, i16 11)		; <i16> [#uses=1]
	store i16 %3, i16* @ss, align 2
	%4 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%5 = call i16 @llvm.atomic.load.add.i16.p0i16(i16* %4, i16 11)		; <i16> [#uses=1]
	store i16 %5, i16* @us, align 2
	%6 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%7 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %6, i32 11)		; <i32> [#uses=1]
	store i32 %7, i32* @si, align 4
	%8 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%9 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %8, i32 11)		; <i32> [#uses=1]
	store i32 %9, i32* @ui, align 4
	%10 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%11 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %10, i32 11)		; <i32> [#uses=1]
	store i32 %11, i32* @sl, align 4
	%12 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%13 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %12, i32 11)		; <i32> [#uses=1]
	store i32 %13, i32* @ul, align 4
	%14 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%15 = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %14, i64 11)		; <i64> [#uses=1]
	store i64 %15, i64* @sll, align 8
	%16 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%17 = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %16, i64 11)		; <i64> [#uses=1]
	store i64 %17, i64* @ull, align 8
	%18 = call i8 @llvm.atomic.load.sub.i8.p0i8(i8* @sc, i8 11)		; <i8> [#uses=1]
	store i8 %18, i8* @sc, align 1
	%19 = call i8 @llvm.atomic.load.sub.i8.p0i8(i8* @uc, i8 11)		; <i8> [#uses=1]
	store i8 %19, i8* @uc, align 1
	%20 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%21 = call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %20, i16 11)		; <i16> [#uses=1]
	store i16 %21, i16* @ss, align 2
	%22 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%23 = call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %22, i16 11)		; <i16> [#uses=1]
	store i16 %23, i16* @us, align 2
	%24 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%25 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %24, i32 11)		; <i32> [#uses=1]
	store i32 %25, i32* @si, align 4
	%26 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%27 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %26, i32 11)		; <i32> [#uses=1]
	store i32 %27, i32* @ui, align 4
	%28 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%29 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %28, i32 11)		; <i32> [#uses=1]
	store i32 %29, i32* @sl, align 4
	%30 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%31 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %30, i32 11)		; <i32> [#uses=1]
	store i32 %31, i32* @ul, align 4
	%32 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%33 = call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %32, i64 11)		; <i64> [#uses=1]
	store i64 %33, i64* @sll, align 8
	%34 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%35 = call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %34, i64 11)		; <i64> [#uses=1]
	store i64 %35, i64* @ull, align 8
	%36 = call i8 @llvm.atomic.load.or.i8.p0i8(i8* @sc, i8 11)		; <i8> [#uses=1]
	store i8 %36, i8* @sc, align 1
	%37 = call i8 @llvm.atomic.load.or.i8.p0i8(i8* @uc, i8 11)		; <i8> [#uses=1]
	store i8 %37, i8* @uc, align 1
	%38 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%39 = call i16 @llvm.atomic.load.or.i16.p0i16(i16* %38, i16 11)		; <i16> [#uses=1]
	store i16 %39, i16* @ss, align 2
	%40 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%41 = call i16 @llvm.atomic.load.or.i16.p0i16(i16* %40, i16 11)		; <i16> [#uses=1]
	store i16 %41, i16* @us, align 2
	%42 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%43 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %42, i32 11)		; <i32> [#uses=1]
	store i32 %43, i32* @si, align 4
	%44 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%45 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %44, i32 11)		; <i32> [#uses=1]
	store i32 %45, i32* @ui, align 4
	%46 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%47 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %46, i32 11)		; <i32> [#uses=1]
	store i32 %47, i32* @sl, align 4
	%48 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%49 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %48, i32 11)		; <i32> [#uses=1]
	store i32 %49, i32* @ul, align 4
	%50 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%51 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %50, i64 11)		; <i64> [#uses=1]
	store i64 %51, i64* @sll, align 8
	%52 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%53 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %52, i64 11)		; <i64> [#uses=1]
	store i64 %53, i64* @ull, align 8
	%54 = call i8 @llvm.atomic.load.xor.i8.p0i8(i8* @sc, i8 11)		; <i8> [#uses=1]
	store i8 %54, i8* @sc, align 1
	%55 = call i8 @llvm.atomic.load.xor.i8.p0i8(i8* @uc, i8 11)		; <i8> [#uses=1]
	store i8 %55, i8* @uc, align 1
	%56 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%57 = call i16 @llvm.atomic.load.xor.i16.p0i16(i16* %56, i16 11)		; <i16> [#uses=1]
	store i16 %57, i16* @ss, align 2
	%58 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%59 = call i16 @llvm.atomic.load.xor.i16.p0i16(i16* %58, i16 11)		; <i16> [#uses=1]
	store i16 %59, i16* @us, align 2
	%60 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%61 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %60, i32 11)		; <i32> [#uses=1]
	store i32 %61, i32* @si, align 4
	%62 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%63 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %62, i32 11)		; <i32> [#uses=1]
	store i32 %63, i32* @ui, align 4
	%64 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%65 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %64, i32 11)		; <i32> [#uses=1]
	store i32 %65, i32* @sl, align 4
	%66 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%67 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %66, i32 11)		; <i32> [#uses=1]
	store i32 %67, i32* @ul, align 4
	%68 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%69 = call i64 @llvm.atomic.load.xor.i64.p0i64(i64* %68, i64 11)		; <i64> [#uses=1]
	store i64 %69, i64* @sll, align 8
	%70 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%71 = call i64 @llvm.atomic.load.xor.i64.p0i64(i64* %70, i64 11)		; <i64> [#uses=1]
	store i64 %71, i64* @ull, align 8
	%72 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @sc, i8 11)		; <i8> [#uses=1]
	store i8 %72, i8* @sc, align 1
	%73 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @uc, i8 11)		; <i8> [#uses=1]
	store i8 %73, i8* @uc, align 1
	%74 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%75 = call i16 @llvm.atomic.load.and.i16.p0i16(i16* %74, i16 11)		; <i16> [#uses=1]
	store i16 %75, i16* @ss, align 2
	%76 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%77 = call i16 @llvm.atomic.load.and.i16.p0i16(i16* %76, i16 11)		; <i16> [#uses=1]
	store i16 %77, i16* @us, align 2
	%78 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%79 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %78, i32 11)		; <i32> [#uses=1]
	store i32 %79, i32* @si, align 4
	%80 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%81 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %80, i32 11)		; <i32> [#uses=1]
	store i32 %81, i32* @ui, align 4
	%82 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%83 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %82, i32 11)		; <i32> [#uses=1]
	store i32 %83, i32* @sl, align 4
	%84 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%85 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %84, i32 11)		; <i32> [#uses=1]
	store i32 %85, i32* @ul, align 4
	%86 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%87 = call i64 @llvm.atomic.load.and.i64.p0i64(i64* %86, i64 11)		; <i64> [#uses=1]
	store i64 %87, i64* @sll, align 8
	%88 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%89 = call i64 @llvm.atomic.load.and.i64.p0i64(i64* %88, i64 11)		; <i64> [#uses=1]
	store i64 %89, i64* @ull, align 8
	%90 = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* @sc, i8 11)		; <i8> [#uses=1]
	store i8 %90, i8* @sc, align 1
	%91 = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* @uc, i8 11)		; <i8> [#uses=1]
	store i8 %91, i8* @uc, align 1
	%92 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%93 = call i16 @llvm.atomic.load.nand.i16.p0i16(i16* %92, i16 11)		; <i16> [#uses=1]
	store i16 %93, i16* @ss, align 2
	%94 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%95 = call i16 @llvm.atomic.load.nand.i16.p0i16(i16* %94, i16 11)		; <i16> [#uses=1]
	store i16 %95, i16* @us, align 2
	%96 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%97 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %96, i32 11)		; <i32> [#uses=1]
	store i32 %97, i32* @si, align 4
	%98 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%99 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %98, i32 11)		; <i32> [#uses=1]
	store i32 %99, i32* @ui, align 4
	%100 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%101 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %100, i32 11)		; <i32> [#uses=1]
	store i32 %101, i32* @sl, align 4
	%102 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%103 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %102, i32 11)		; <i32> [#uses=1]
	store i32 %103, i32* @ul, align 4
	%104 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%105 = call i64 @llvm.atomic.load.nand.i64.p0i64(i64* %104, i64 11)		; <i64> [#uses=1]
	store i64 %105, i64* @sll, align 8
	%106 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%107 = call i64 @llvm.atomic.load.nand.i64.p0i64(i64* %106, i64 11)		; <i64> [#uses=1]
	store i64 %107, i64* @ull, align 8
	br label %return

return:		; preds = %entry
	ret void
}

define void @test_op_and_fetch() nounwind {
entry:
	%0 = load i8* @uc, align 1		; <i8> [#uses=1]
	%1 = zext i8 %0 to i32		; <i32> [#uses=1]
	%2 = trunc i32 %1 to i8		; <i8> [#uses=2]
	%3 = call i8 @llvm.atomic.load.add.i8.p0i8(i8* @sc, i8 %2)		; <i8> [#uses=1]
	%4 = add i8 %3, %2		; <i8> [#uses=1]
	store i8 %4, i8* @sc, align 1
	%5 = load i8* @uc, align 1		; <i8> [#uses=1]
	%6 = zext i8 %5 to i32		; <i32> [#uses=1]
	%7 = trunc i32 %6 to i8		; <i8> [#uses=2]
	%8 = call i8 @llvm.atomic.load.add.i8.p0i8(i8* @uc, i8 %7)		; <i8> [#uses=1]
	%9 = add i8 %8, %7		; <i8> [#uses=1]
	store i8 %9, i8* @uc, align 1
	%10 = load i8* @uc, align 1		; <i8> [#uses=1]
	%11 = zext i8 %10 to i32		; <i32> [#uses=1]
	%12 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%13 = trunc i32 %11 to i16		; <i16> [#uses=2]
	%14 = call i16 @llvm.atomic.load.add.i16.p0i16(i16* %12, i16 %13)		; <i16> [#uses=1]
	%15 = add i16 %14, %13		; <i16> [#uses=1]
	store i16 %15, i16* @ss, align 2
	%16 = load i8* @uc, align 1		; <i8> [#uses=1]
	%17 = zext i8 %16 to i32		; <i32> [#uses=1]
	%18 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%19 = trunc i32 %17 to i16		; <i16> [#uses=2]
	%20 = call i16 @llvm.atomic.load.add.i16.p0i16(i16* %18, i16 %19)		; <i16> [#uses=1]
	%21 = add i16 %20, %19		; <i16> [#uses=1]
	store i16 %21, i16* @us, align 2
	%22 = load i8* @uc, align 1		; <i8> [#uses=1]
	%23 = zext i8 %22 to i32		; <i32> [#uses=2]
	%24 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%25 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %24, i32 %23)		; <i32> [#uses=1]
	%26 = add i32 %25, %23		; <i32> [#uses=1]
	store i32 %26, i32* @si, align 4
	%27 = load i8* @uc, align 1		; <i8> [#uses=1]
	%28 = zext i8 %27 to i32		; <i32> [#uses=2]
	%29 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%30 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %29, i32 %28)		; <i32> [#uses=1]
	%31 = add i32 %30, %28		; <i32> [#uses=1]
	store i32 %31, i32* @ui, align 4
	%32 = load i8* @uc, align 1		; <i8> [#uses=1]
	%33 = zext i8 %32 to i32		; <i32> [#uses=2]
	%34 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%35 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %34, i32 %33)		; <i32> [#uses=1]
	%36 = add i32 %35, %33		; <i32> [#uses=1]
	store i32 %36, i32* @sl, align 4
	%37 = load i8* @uc, align 1		; <i8> [#uses=1]
	%38 = zext i8 %37 to i32		; <i32> [#uses=2]
	%39 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%40 = call i32 @llvm.atomic.load.add.i32.p0i32(i32* %39, i32 %38)		; <i32> [#uses=1]
	%41 = add i32 %40, %38		; <i32> [#uses=1]
	store i32 %41, i32* @ul, align 4
	%42 = load i8* @uc, align 1		; <i8> [#uses=1]
	%43 = zext i8 %42 to i64		; <i64> [#uses=2]
	%44 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%45 = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %44, i64 %43)		; <i64> [#uses=1]
	%46 = add i64 %45, %43		; <i64> [#uses=1]
	store i64 %46, i64* @sll, align 8
	%47 = load i8* @uc, align 1		; <i8> [#uses=1]
	%48 = zext i8 %47 to i64		; <i64> [#uses=2]
	%49 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%50 = call i64 @llvm.atomic.load.add.i64.p0i64(i64* %49, i64 %48)		; <i64> [#uses=1]
	%51 = add i64 %50, %48		; <i64> [#uses=1]
	store i64 %51, i64* @ull, align 8
	%52 = load i8* @uc, align 1		; <i8> [#uses=1]
	%53 = zext i8 %52 to i32		; <i32> [#uses=1]
	%54 = trunc i32 %53 to i8		; <i8> [#uses=2]
	%55 = call i8 @llvm.atomic.load.sub.i8.p0i8(i8* @sc, i8 %54)		; <i8> [#uses=1]
	%56 = sub i8 %55, %54		; <i8> [#uses=1]
	store i8 %56, i8* @sc, align 1
	%57 = load i8* @uc, align 1		; <i8> [#uses=1]
	%58 = zext i8 %57 to i32		; <i32> [#uses=1]
	%59 = trunc i32 %58 to i8		; <i8> [#uses=2]
	%60 = call i8 @llvm.atomic.load.sub.i8.p0i8(i8* @uc, i8 %59)		; <i8> [#uses=1]
	%61 = sub i8 %60, %59		; <i8> [#uses=1]
	store i8 %61, i8* @uc, align 1
	%62 = load i8* @uc, align 1		; <i8> [#uses=1]
	%63 = zext i8 %62 to i32		; <i32> [#uses=1]
	%64 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%65 = trunc i32 %63 to i16		; <i16> [#uses=2]
	%66 = call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %64, i16 %65)		; <i16> [#uses=1]
	%67 = sub i16 %66, %65		; <i16> [#uses=1]
	store i16 %67, i16* @ss, align 2
	%68 = load i8* @uc, align 1		; <i8> [#uses=1]
	%69 = zext i8 %68 to i32		; <i32> [#uses=1]
	%70 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%71 = trunc i32 %69 to i16		; <i16> [#uses=2]
	%72 = call i16 @llvm.atomic.load.sub.i16.p0i16(i16* %70, i16 %71)		; <i16> [#uses=1]
	%73 = sub i16 %72, %71		; <i16> [#uses=1]
	store i16 %73, i16* @us, align 2
	%74 = load i8* @uc, align 1		; <i8> [#uses=1]
	%75 = zext i8 %74 to i32		; <i32> [#uses=2]
	%76 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%77 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %76, i32 %75)		; <i32> [#uses=1]
	%78 = sub i32 %77, %75		; <i32> [#uses=1]
	store i32 %78, i32* @si, align 4
	%79 = load i8* @uc, align 1		; <i8> [#uses=1]
	%80 = zext i8 %79 to i32		; <i32> [#uses=2]
	%81 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%82 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %81, i32 %80)		; <i32> [#uses=1]
	%83 = sub i32 %82, %80		; <i32> [#uses=1]
	store i32 %83, i32* @ui, align 4
	%84 = load i8* @uc, align 1		; <i8> [#uses=1]
	%85 = zext i8 %84 to i32		; <i32> [#uses=2]
	%86 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%87 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %86, i32 %85)		; <i32> [#uses=1]
	%88 = sub i32 %87, %85		; <i32> [#uses=1]
	store i32 %88, i32* @sl, align 4
	%89 = load i8* @uc, align 1		; <i8> [#uses=1]
	%90 = zext i8 %89 to i32		; <i32> [#uses=2]
	%91 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%92 = call i32 @llvm.atomic.load.sub.i32.p0i32(i32* %91, i32 %90)		; <i32> [#uses=1]
	%93 = sub i32 %92, %90		; <i32> [#uses=1]
	store i32 %93, i32* @ul, align 4
	%94 = load i8* @uc, align 1		; <i8> [#uses=1]
	%95 = zext i8 %94 to i64		; <i64> [#uses=2]
	%96 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%97 = call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %96, i64 %95)		; <i64> [#uses=1]
	%98 = sub i64 %97, %95		; <i64> [#uses=1]
	store i64 %98, i64* @sll, align 8
	%99 = load i8* @uc, align 1		; <i8> [#uses=1]
	%100 = zext i8 %99 to i64		; <i64> [#uses=2]
	%101 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%102 = call i64 @llvm.atomic.load.sub.i64.p0i64(i64* %101, i64 %100)		; <i64> [#uses=1]
	%103 = sub i64 %102, %100		; <i64> [#uses=1]
	store i64 %103, i64* @ull, align 8
	%104 = load i8* @uc, align 1		; <i8> [#uses=1]
	%105 = zext i8 %104 to i32		; <i32> [#uses=1]
	%106 = trunc i32 %105 to i8		; <i8> [#uses=2]
	%107 = call i8 @llvm.atomic.load.or.i8.p0i8(i8* @sc, i8 %106)		; <i8> [#uses=1]
	%108 = or i8 %107, %106		; <i8> [#uses=1]
	store i8 %108, i8* @sc, align 1
	%109 = load i8* @uc, align 1		; <i8> [#uses=1]
	%110 = zext i8 %109 to i32		; <i32> [#uses=1]
	%111 = trunc i32 %110 to i8		; <i8> [#uses=2]
	%112 = call i8 @llvm.atomic.load.or.i8.p0i8(i8* @uc, i8 %111)		; <i8> [#uses=1]
	%113 = or i8 %112, %111		; <i8> [#uses=1]
	store i8 %113, i8* @uc, align 1
	%114 = load i8* @uc, align 1		; <i8> [#uses=1]
	%115 = zext i8 %114 to i32		; <i32> [#uses=1]
	%116 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%117 = trunc i32 %115 to i16		; <i16> [#uses=2]
	%118 = call i16 @llvm.atomic.load.or.i16.p0i16(i16* %116, i16 %117)		; <i16> [#uses=1]
	%119 = or i16 %118, %117		; <i16> [#uses=1]
	store i16 %119, i16* @ss, align 2
	%120 = load i8* @uc, align 1		; <i8> [#uses=1]
	%121 = zext i8 %120 to i32		; <i32> [#uses=1]
	%122 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%123 = trunc i32 %121 to i16		; <i16> [#uses=2]
	%124 = call i16 @llvm.atomic.load.or.i16.p0i16(i16* %122, i16 %123)		; <i16> [#uses=1]
	%125 = or i16 %124, %123		; <i16> [#uses=1]
	store i16 %125, i16* @us, align 2
	%126 = load i8* @uc, align 1		; <i8> [#uses=1]
	%127 = zext i8 %126 to i32		; <i32> [#uses=2]
	%128 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%129 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %128, i32 %127)		; <i32> [#uses=1]
	%130 = or i32 %129, %127		; <i32> [#uses=1]
	store i32 %130, i32* @si, align 4
	%131 = load i8* @uc, align 1		; <i8> [#uses=1]
	%132 = zext i8 %131 to i32		; <i32> [#uses=2]
	%133 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%134 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %133, i32 %132)		; <i32> [#uses=1]
	%135 = or i32 %134, %132		; <i32> [#uses=1]
	store i32 %135, i32* @ui, align 4
	%136 = load i8* @uc, align 1		; <i8> [#uses=1]
	%137 = zext i8 %136 to i32		; <i32> [#uses=2]
	%138 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%139 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %138, i32 %137)		; <i32> [#uses=1]
	%140 = or i32 %139, %137		; <i32> [#uses=1]
	store i32 %140, i32* @sl, align 4
	%141 = load i8* @uc, align 1		; <i8> [#uses=1]
	%142 = zext i8 %141 to i32		; <i32> [#uses=2]
	%143 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%144 = call i32 @llvm.atomic.load.or.i32.p0i32(i32* %143, i32 %142)		; <i32> [#uses=1]
	%145 = or i32 %144, %142		; <i32> [#uses=1]
	store i32 %145, i32* @ul, align 4
	%146 = load i8* @uc, align 1		; <i8> [#uses=1]
	%147 = zext i8 %146 to i64		; <i64> [#uses=2]
	%148 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%149 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %148, i64 %147)		; <i64> [#uses=1]
	%150 = or i64 %149, %147		; <i64> [#uses=1]
	store i64 %150, i64* @sll, align 8
	%151 = load i8* @uc, align 1		; <i8> [#uses=1]
	%152 = zext i8 %151 to i64		; <i64> [#uses=2]
	%153 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%154 = call i64 @llvm.atomic.load.or.i64.p0i64(i64* %153, i64 %152)		; <i64> [#uses=1]
	%155 = or i64 %154, %152		; <i64> [#uses=1]
	store i64 %155, i64* @ull, align 8
	%156 = load i8* @uc, align 1		; <i8> [#uses=1]
	%157 = zext i8 %156 to i32		; <i32> [#uses=1]
	%158 = trunc i32 %157 to i8		; <i8> [#uses=2]
	%159 = call i8 @llvm.atomic.load.xor.i8.p0i8(i8* @sc, i8 %158)		; <i8> [#uses=1]
	%160 = xor i8 %159, %158		; <i8> [#uses=1]
	store i8 %160, i8* @sc, align 1
	%161 = load i8* @uc, align 1		; <i8> [#uses=1]
	%162 = zext i8 %161 to i32		; <i32> [#uses=1]
	%163 = trunc i32 %162 to i8		; <i8> [#uses=2]
	%164 = call i8 @llvm.atomic.load.xor.i8.p0i8(i8* @uc, i8 %163)		; <i8> [#uses=1]
	%165 = xor i8 %164, %163		; <i8> [#uses=1]
	store i8 %165, i8* @uc, align 1
	%166 = load i8* @uc, align 1		; <i8> [#uses=1]
	%167 = zext i8 %166 to i32		; <i32> [#uses=1]
	%168 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%169 = trunc i32 %167 to i16		; <i16> [#uses=2]
	%170 = call i16 @llvm.atomic.load.xor.i16.p0i16(i16* %168, i16 %169)		; <i16> [#uses=1]
	%171 = xor i16 %170, %169		; <i16> [#uses=1]
	store i16 %171, i16* @ss, align 2
	%172 = load i8* @uc, align 1		; <i8> [#uses=1]
	%173 = zext i8 %172 to i32		; <i32> [#uses=1]
	%174 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%175 = trunc i32 %173 to i16		; <i16> [#uses=2]
	%176 = call i16 @llvm.atomic.load.xor.i16.p0i16(i16* %174, i16 %175)		; <i16> [#uses=1]
	%177 = xor i16 %176, %175		; <i16> [#uses=1]
	store i16 %177, i16* @us, align 2
	%178 = load i8* @uc, align 1		; <i8> [#uses=1]
	%179 = zext i8 %178 to i32		; <i32> [#uses=2]
	%180 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%181 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %180, i32 %179)		; <i32> [#uses=1]
	%182 = xor i32 %181, %179		; <i32> [#uses=1]
	store i32 %182, i32* @si, align 4
	%183 = load i8* @uc, align 1		; <i8> [#uses=1]
	%184 = zext i8 %183 to i32		; <i32> [#uses=2]
	%185 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%186 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %185, i32 %184)		; <i32> [#uses=1]
	%187 = xor i32 %186, %184		; <i32> [#uses=1]
	store i32 %187, i32* @ui, align 4
	%188 = load i8* @uc, align 1		; <i8> [#uses=1]
	%189 = zext i8 %188 to i32		; <i32> [#uses=2]
	%190 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%191 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %190, i32 %189)		; <i32> [#uses=1]
	%192 = xor i32 %191, %189		; <i32> [#uses=1]
	store i32 %192, i32* @sl, align 4
	%193 = load i8* @uc, align 1		; <i8> [#uses=1]
	%194 = zext i8 %193 to i32		; <i32> [#uses=2]
	%195 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%196 = call i32 @llvm.atomic.load.xor.i32.p0i32(i32* %195, i32 %194)		; <i32> [#uses=1]
	%197 = xor i32 %196, %194		; <i32> [#uses=1]
	store i32 %197, i32* @ul, align 4
	%198 = load i8* @uc, align 1		; <i8> [#uses=1]
	%199 = zext i8 %198 to i64		; <i64> [#uses=2]
	%200 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%201 = call i64 @llvm.atomic.load.xor.i64.p0i64(i64* %200, i64 %199)		; <i64> [#uses=1]
	%202 = xor i64 %201, %199		; <i64> [#uses=1]
	store i64 %202, i64* @sll, align 8
	%203 = load i8* @uc, align 1		; <i8> [#uses=1]
	%204 = zext i8 %203 to i64		; <i64> [#uses=2]
	%205 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%206 = call i64 @llvm.atomic.load.xor.i64.p0i64(i64* %205, i64 %204)		; <i64> [#uses=1]
	%207 = xor i64 %206, %204		; <i64> [#uses=1]
	store i64 %207, i64* @ull, align 8
	%208 = load i8* @uc, align 1		; <i8> [#uses=1]
	%209 = zext i8 %208 to i32		; <i32> [#uses=1]
	%210 = trunc i32 %209 to i8		; <i8> [#uses=2]
	%211 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @sc, i8 %210)		; <i8> [#uses=1]
	%212 = and i8 %211, %210		; <i8> [#uses=1]
	store i8 %212, i8* @sc, align 1
	%213 = load i8* @uc, align 1		; <i8> [#uses=1]
	%214 = zext i8 %213 to i32		; <i32> [#uses=1]
	%215 = trunc i32 %214 to i8		; <i8> [#uses=2]
	%216 = call i8 @llvm.atomic.load.and.i8.p0i8(i8* @uc, i8 %215)		; <i8> [#uses=1]
	%217 = and i8 %216, %215		; <i8> [#uses=1]
	store i8 %217, i8* @uc, align 1
	%218 = load i8* @uc, align 1		; <i8> [#uses=1]
	%219 = zext i8 %218 to i32		; <i32> [#uses=1]
	%220 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%221 = trunc i32 %219 to i16		; <i16> [#uses=2]
	%222 = call i16 @llvm.atomic.load.and.i16.p0i16(i16* %220, i16 %221)		; <i16> [#uses=1]
	%223 = and i16 %222, %221		; <i16> [#uses=1]
	store i16 %223, i16* @ss, align 2
	%224 = load i8* @uc, align 1		; <i8> [#uses=1]
	%225 = zext i8 %224 to i32		; <i32> [#uses=1]
	%226 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%227 = trunc i32 %225 to i16		; <i16> [#uses=2]
	%228 = call i16 @llvm.atomic.load.and.i16.p0i16(i16* %226, i16 %227)		; <i16> [#uses=1]
	%229 = and i16 %228, %227		; <i16> [#uses=1]
	store i16 %229, i16* @us, align 2
	%230 = load i8* @uc, align 1		; <i8> [#uses=1]
	%231 = zext i8 %230 to i32		; <i32> [#uses=2]
	%232 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%233 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %232, i32 %231)		; <i32> [#uses=1]
	%234 = and i32 %233, %231		; <i32> [#uses=1]
	store i32 %234, i32* @si, align 4
	%235 = load i8* @uc, align 1		; <i8> [#uses=1]
	%236 = zext i8 %235 to i32		; <i32> [#uses=2]
	%237 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%238 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %237, i32 %236)		; <i32> [#uses=1]
	%239 = and i32 %238, %236		; <i32> [#uses=1]
	store i32 %239, i32* @ui, align 4
	%240 = load i8* @uc, align 1		; <i8> [#uses=1]
	%241 = zext i8 %240 to i32		; <i32> [#uses=2]
	%242 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%243 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %242, i32 %241)		; <i32> [#uses=1]
	%244 = and i32 %243, %241		; <i32> [#uses=1]
	store i32 %244, i32* @sl, align 4
	%245 = load i8* @uc, align 1		; <i8> [#uses=1]
	%246 = zext i8 %245 to i32		; <i32> [#uses=2]
	%247 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%248 = call i32 @llvm.atomic.load.and.i32.p0i32(i32* %247, i32 %246)		; <i32> [#uses=1]
	%249 = and i32 %248, %246		; <i32> [#uses=1]
	store i32 %249, i32* @ul, align 4
	%250 = load i8* @uc, align 1		; <i8> [#uses=1]
	%251 = zext i8 %250 to i64		; <i64> [#uses=2]
	%252 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%253 = call i64 @llvm.atomic.load.and.i64.p0i64(i64* %252, i64 %251)		; <i64> [#uses=1]
	%254 = and i64 %253, %251		; <i64> [#uses=1]
	store i64 %254, i64* @sll, align 8
	%255 = load i8* @uc, align 1		; <i8> [#uses=1]
	%256 = zext i8 %255 to i64		; <i64> [#uses=2]
	%257 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%258 = call i64 @llvm.atomic.load.and.i64.p0i64(i64* %257, i64 %256)		; <i64> [#uses=1]
	%259 = and i64 %258, %256		; <i64> [#uses=1]
	store i64 %259, i64* @ull, align 8
	%260 = load i8* @uc, align 1		; <i8> [#uses=1]
	%261 = zext i8 %260 to i32		; <i32> [#uses=1]
	%262 = trunc i32 %261 to i8		; <i8> [#uses=2]
	%263 = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* @sc, i8 %262)		; <i8> [#uses=1]
	%264 = xor i8 %263, -1		; <i8> [#uses=1]
	%265 = and i8 %264, %262		; <i8> [#uses=1]
	store i8 %265, i8* @sc, align 1
	%266 = load i8* @uc, align 1		; <i8> [#uses=1]
	%267 = zext i8 %266 to i32		; <i32> [#uses=1]
	%268 = trunc i32 %267 to i8		; <i8> [#uses=2]
	%269 = call i8 @llvm.atomic.load.nand.i8.p0i8(i8* @uc, i8 %268)		; <i8> [#uses=1]
	%270 = xor i8 %269, -1		; <i8> [#uses=1]
	%271 = and i8 %270, %268		; <i8> [#uses=1]
	store i8 %271, i8* @uc, align 1
	%272 = load i8* @uc, align 1		; <i8> [#uses=1]
	%273 = zext i8 %272 to i32		; <i32> [#uses=1]
	%274 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%275 = trunc i32 %273 to i16		; <i16> [#uses=2]
	%276 = call i16 @llvm.atomic.load.nand.i16.p0i16(i16* %274, i16 %275)		; <i16> [#uses=1]
	%277 = xor i16 %276, -1		; <i16> [#uses=1]
	%278 = and i16 %277, %275		; <i16> [#uses=1]
	store i16 %278, i16* @ss, align 2
	%279 = load i8* @uc, align 1		; <i8> [#uses=1]
	%280 = zext i8 %279 to i32		; <i32> [#uses=1]
	%281 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%282 = trunc i32 %280 to i16		; <i16> [#uses=2]
	%283 = call i16 @llvm.atomic.load.nand.i16.p0i16(i16* %281, i16 %282)		; <i16> [#uses=1]
	%284 = xor i16 %283, -1		; <i16> [#uses=1]
	%285 = and i16 %284, %282		; <i16> [#uses=1]
	store i16 %285, i16* @us, align 2
	%286 = load i8* @uc, align 1		; <i8> [#uses=1]
	%287 = zext i8 %286 to i32		; <i32> [#uses=2]
	%288 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%289 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %288, i32 %287)		; <i32> [#uses=1]
	%290 = xor i32 %289, -1		; <i32> [#uses=1]
	%291 = and i32 %290, %287		; <i32> [#uses=1]
	store i32 %291, i32* @si, align 4
	%292 = load i8* @uc, align 1		; <i8> [#uses=1]
	%293 = zext i8 %292 to i32		; <i32> [#uses=2]
	%294 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%295 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %294, i32 %293)		; <i32> [#uses=1]
	%296 = xor i32 %295, -1		; <i32> [#uses=1]
	%297 = and i32 %296, %293		; <i32> [#uses=1]
	store i32 %297, i32* @ui, align 4
	%298 = load i8* @uc, align 1		; <i8> [#uses=1]
	%299 = zext i8 %298 to i32		; <i32> [#uses=2]
	%300 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%301 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %300, i32 %299)		; <i32> [#uses=1]
	%302 = xor i32 %301, -1		; <i32> [#uses=1]
	%303 = and i32 %302, %299		; <i32> [#uses=1]
	store i32 %303, i32* @sl, align 4
	%304 = load i8* @uc, align 1		; <i8> [#uses=1]
	%305 = zext i8 %304 to i32		; <i32> [#uses=2]
	%306 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%307 = call i32 @llvm.atomic.load.nand.i32.p0i32(i32* %306, i32 %305)		; <i32> [#uses=1]
	%308 = xor i32 %307, -1		; <i32> [#uses=1]
	%309 = and i32 %308, %305		; <i32> [#uses=1]
	store i32 %309, i32* @ul, align 4
	%310 = load i8* @uc, align 1		; <i8> [#uses=1]
	%311 = zext i8 %310 to i64		; <i64> [#uses=2]
	%312 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	%313 = call i64 @llvm.atomic.load.nand.i64.p0i64(i64* %312, i64 %311)		; <i64> [#uses=1]
	%314 = xor i64 %313, -1		; <i64> [#uses=1]
	%315 = and i64 %314, %311		; <i64> [#uses=1]
	store i64 %315, i64* @sll, align 8
	%316 = load i8* @uc, align 1		; <i8> [#uses=1]
	%317 = zext i8 %316 to i64		; <i64> [#uses=2]
	%318 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	%319 = call i64 @llvm.atomic.load.nand.i64.p0i64(i64* %318, i64 %317)		; <i64> [#uses=1]
	%320 = xor i64 %319, -1		; <i64> [#uses=1]
	%321 = and i64 %320, %317		; <i64> [#uses=1]
	store i64 %321, i64* @ull, align 8
	br label %return

return:		; preds = %entry
	ret void
}

define void @test_compare_and_swap() nounwind {
entry:
	%0 = load i8* @sc, align 1		; <i8> [#uses=1]
	%1 = zext i8 %0 to i32		; <i32> [#uses=1]
	%2 = load i8* @uc, align 1		; <i8> [#uses=1]
	%3 = zext i8 %2 to i32		; <i32> [#uses=1]
	%4 = trunc i32 %3 to i8		; <i8> [#uses=1]
	%5 = trunc i32 %1 to i8		; <i8> [#uses=1]
	%6 = call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* @sc, i8 %4, i8 %5)		; <i8> [#uses=1]
	store i8 %6, i8* @sc, align 1
	%7 = load i8* @sc, align 1		; <i8> [#uses=1]
	%8 = zext i8 %7 to i32		; <i32> [#uses=1]
	%9 = load i8* @uc, align 1		; <i8> [#uses=1]
	%10 = zext i8 %9 to i32		; <i32> [#uses=1]
	%11 = trunc i32 %10 to i8		; <i8> [#uses=1]
	%12 = trunc i32 %8 to i8		; <i8> [#uses=1]
	%13 = call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* @uc, i8 %11, i8 %12)		; <i8> [#uses=1]
	store i8 %13, i8* @uc, align 1
	%14 = load i8* @sc, align 1		; <i8> [#uses=1]
	%15 = sext i8 %14 to i16		; <i16> [#uses=1]
	%16 = zext i16 %15 to i32		; <i32> [#uses=1]
	%17 = load i8* @uc, align 1		; <i8> [#uses=1]
	%18 = zext i8 %17 to i32		; <i32> [#uses=1]
	%19 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%20 = trunc i32 %18 to i16		; <i16> [#uses=1]
	%21 = trunc i32 %16 to i16		; <i16> [#uses=1]
	%22 = call i16 @llvm.atomic.cmp.swap.i16.p0i16(i16* %19, i16 %20, i16 %21)		; <i16> [#uses=1]
	store i16 %22, i16* @ss, align 2
	%23 = load i8* @sc, align 1		; <i8> [#uses=1]
	%24 = sext i8 %23 to i16		; <i16> [#uses=1]
	%25 = zext i16 %24 to i32		; <i32> [#uses=1]
	%26 = load i8* @uc, align 1		; <i8> [#uses=1]
	%27 = zext i8 %26 to i32		; <i32> [#uses=1]
	%28 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%29 = trunc i32 %27 to i16		; <i16> [#uses=1]
	%30 = trunc i32 %25 to i16		; <i16> [#uses=1]
	%31 = call i16 @llvm.atomic.cmp.swap.i16.p0i16(i16* %28, i16 %29, i16 %30)		; <i16> [#uses=1]
	store i16 %31, i16* @us, align 2
	%32 = load i8* @sc, align 1		; <i8> [#uses=1]
	%33 = sext i8 %32 to i32		; <i32> [#uses=1]
	%34 = load i8* @uc, align 1		; <i8> [#uses=1]
	%35 = zext i8 %34 to i32		; <i32> [#uses=1]
	%36 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%37 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %36, i32 %35, i32 %33)		; <i32> [#uses=1]
	store i32 %37, i32* @si, align 4
	%38 = load i8* @sc, align 1		; <i8> [#uses=1]
	%39 = sext i8 %38 to i32		; <i32> [#uses=1]
	%40 = load i8* @uc, align 1		; <i8> [#uses=1]
	%41 = zext i8 %40 to i32		; <i32> [#uses=1]
	%42 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%43 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %42, i32 %41, i32 %39)		; <i32> [#uses=1]
	store i32 %43, i32* @ui, align 4
	%44 = load i8* @sc, align 1		; <i8> [#uses=1]
	%45 = sext i8 %44 to i32		; <i32> [#uses=1]
	%46 = load i8* @uc, align 1		; <i8> [#uses=1]
	%47 = zext i8 %46 to i32		; <i32> [#uses=1]
	%48 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%49 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %48, i32 %47, i32 %45)		; <i32> [#uses=1]
	store i32 %49, i32* @sl, align 4
	%50 = load i8* @sc, align 1		; <i8> [#uses=1]
	%51 = sext i8 %50 to i32		; <i32> [#uses=1]
	%52 = load i8* @uc, align 1		; <i8> [#uses=1]
	%53 = zext i8 %52 to i32		; <i32> [#uses=1]
	%54 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%55 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %54, i32 %53, i32 %51)		; <i32> [#uses=1]
	store i32 %55, i32* @ul, align 4
	%56 = load i8* @sc, align 1		; <i8> [#uses=1]
	%57 = zext i8 %56 to i32		; <i32> [#uses=1]
	%58 = load i8* @uc, align 1		; <i8> [#uses=1]
	%59 = zext i8 %58 to i32		; <i32> [#uses=1]
	%60 = trunc i32 %59 to i8		; <i8> [#uses=2]
	%61 = trunc i32 %57 to i8		; <i8> [#uses=1]
	%62 = call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* @sc, i8 %60, i8 %61)		; <i8> [#uses=1]
	%63 = icmp eq i8 %62, %60		; <i1> [#uses=1]
	%64 = zext i1 %63 to i8		; <i8> [#uses=1]
	%65 = zext i8 %64 to i32		; <i32> [#uses=1]
	store i32 %65, i32* @ui, align 4
	%66 = load i8* @sc, align 1		; <i8> [#uses=1]
	%67 = zext i8 %66 to i32		; <i32> [#uses=1]
	%68 = load i8* @uc, align 1		; <i8> [#uses=1]
	%69 = zext i8 %68 to i32		; <i32> [#uses=1]
	%70 = trunc i32 %69 to i8		; <i8> [#uses=2]
	%71 = trunc i32 %67 to i8		; <i8> [#uses=1]
	%72 = call i8 @llvm.atomic.cmp.swap.i8.p0i8(i8* @uc, i8 %70, i8 %71)		; <i8> [#uses=1]
	%73 = icmp eq i8 %72, %70		; <i1> [#uses=1]
	%74 = zext i1 %73 to i8		; <i8> [#uses=1]
	%75 = zext i8 %74 to i32		; <i32> [#uses=1]
	store i32 %75, i32* @ui, align 4
	%76 = load i8* @sc, align 1		; <i8> [#uses=1]
	%77 = sext i8 %76 to i16		; <i16> [#uses=1]
	%78 = zext i16 %77 to i32		; <i32> [#uses=1]
	%79 = load i8* @uc, align 1		; <i8> [#uses=1]
	%80 = zext i8 %79 to i32		; <i32> [#uses=1]
	%81 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%82 = trunc i32 %80 to i16		; <i16> [#uses=2]
	%83 = trunc i32 %78 to i16		; <i16> [#uses=1]
	%84 = call i16 @llvm.atomic.cmp.swap.i16.p0i16(i16* %81, i16 %82, i16 %83)		; <i16> [#uses=1]
	%85 = icmp eq i16 %84, %82		; <i1> [#uses=1]
	%86 = zext i1 %85 to i8		; <i8> [#uses=1]
	%87 = zext i8 %86 to i32		; <i32> [#uses=1]
	store i32 %87, i32* @ui, align 4
	%88 = load i8* @sc, align 1		; <i8> [#uses=1]
	%89 = sext i8 %88 to i16		; <i16> [#uses=1]
	%90 = zext i16 %89 to i32		; <i32> [#uses=1]
	%91 = load i8* @uc, align 1		; <i8> [#uses=1]
	%92 = zext i8 %91 to i32		; <i32> [#uses=1]
	%93 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%94 = trunc i32 %92 to i16		; <i16> [#uses=2]
	%95 = trunc i32 %90 to i16		; <i16> [#uses=1]
	%96 = call i16 @llvm.atomic.cmp.swap.i16.p0i16(i16* %93, i16 %94, i16 %95)		; <i16> [#uses=1]
	%97 = icmp eq i16 %96, %94		; <i1> [#uses=1]
	%98 = zext i1 %97 to i8		; <i8> [#uses=1]
	%99 = zext i8 %98 to i32		; <i32> [#uses=1]
	store i32 %99, i32* @ui, align 4
	%100 = load i8* @sc, align 1		; <i8> [#uses=1]
	%101 = sext i8 %100 to i32		; <i32> [#uses=1]
	%102 = load i8* @uc, align 1		; <i8> [#uses=1]
	%103 = zext i8 %102 to i32		; <i32> [#uses=2]
	%104 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%105 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %104, i32 %103, i32 %101)		; <i32> [#uses=1]
	%106 = icmp eq i32 %105, %103		; <i1> [#uses=1]
	%107 = zext i1 %106 to i8		; <i8> [#uses=1]
	%108 = zext i8 %107 to i32		; <i32> [#uses=1]
	store i32 %108, i32* @ui, align 4
	%109 = load i8* @sc, align 1		; <i8> [#uses=1]
	%110 = sext i8 %109 to i32		; <i32> [#uses=1]
	%111 = load i8* @uc, align 1		; <i8> [#uses=1]
	%112 = zext i8 %111 to i32		; <i32> [#uses=2]
	%113 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%114 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %113, i32 %112, i32 %110)		; <i32> [#uses=1]
	%115 = icmp eq i32 %114, %112		; <i1> [#uses=1]
	%116 = zext i1 %115 to i8		; <i8> [#uses=1]
	%117 = zext i8 %116 to i32		; <i32> [#uses=1]
	store i32 %117, i32* @ui, align 4
	%118 = load i8* @sc, align 1		; <i8> [#uses=1]
	%119 = sext i8 %118 to i32		; <i32> [#uses=1]
	%120 = load i8* @uc, align 1		; <i8> [#uses=1]
	%121 = zext i8 %120 to i32		; <i32> [#uses=2]
	%122 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%123 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %122, i32 %121, i32 %119)		; <i32> [#uses=1]
	%124 = icmp eq i32 %123, %121		; <i1> [#uses=1]
	%125 = zext i1 %124 to i8		; <i8> [#uses=1]
	%126 = zext i8 %125 to i32		; <i32> [#uses=1]
	store i32 %126, i32* @ui, align 4
	%127 = load i8* @sc, align 1		; <i8> [#uses=1]
	%128 = sext i8 %127 to i32		; <i32> [#uses=1]
	%129 = load i8* @uc, align 1		; <i8> [#uses=1]
	%130 = zext i8 %129 to i32		; <i32> [#uses=2]
	%131 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%132 = call i32 @llvm.atomic.cmp.swap.i32.p0i32(i32* %131, i32 %130, i32 %128)		; <i32> [#uses=1]
	%133 = icmp eq i32 %132, %130		; <i1> [#uses=1]
	%134 = zext i1 %133 to i8		; <i8> [#uses=1]
	%135 = zext i8 %134 to i32		; <i32> [#uses=1]
	store i32 %135, i32* @ui, align 4
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.cmp.swap.i8.p0i8(i8*, i8, i8) nounwind

declare i16 @llvm.atomic.cmp.swap.i16.p0i16(i16*, i16, i16) nounwind

declare i32 @llvm.atomic.cmp.swap.i32.p0i32(i32*, i32, i32) nounwind

define void @test_lock() nounwind {
entry:
	%0 = call i8 @llvm.atomic.swap.i8.p0i8(i8* @sc, i8 1)		; <i8> [#uses=1]
	store i8 %0, i8* @sc, align 1
	%1 = call i8 @llvm.atomic.swap.i8.p0i8(i8* @uc, i8 1)		; <i8> [#uses=1]
	store i8 %1, i8* @uc, align 1
	%2 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	%3 = call i16 @llvm.atomic.swap.i16.p0i16(i16* %2, i16 1)		; <i16> [#uses=1]
	store i16 %3, i16* @ss, align 2
	%4 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	%5 = call i16 @llvm.atomic.swap.i16.p0i16(i16* %4, i16 1)		; <i16> [#uses=1]
	store i16 %5, i16* @us, align 2
	%6 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	%7 = call i32 @llvm.atomic.swap.i32.p0i32(i32* %6, i32 1)		; <i32> [#uses=1]
	store i32 %7, i32* @si, align 4
	%8 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	%9 = call i32 @llvm.atomic.swap.i32.p0i32(i32* %8, i32 1)		; <i32> [#uses=1]
	store i32 %9, i32* @ui, align 4
	%10 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	%11 = call i32 @llvm.atomic.swap.i32.p0i32(i32* %10, i32 1)		; <i32> [#uses=1]
	store i32 %11, i32* @sl, align 4
	%12 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	%13 = call i32 @llvm.atomic.swap.i32.p0i32(i32* %12, i32 1)		; <i32> [#uses=1]
	store i32 %13, i32* @ul, align 4
	call void @llvm.memory.barrier(i1 true, i1 true, i1 true, i1 true, i1 false)
	volatile store i8 0, i8* @sc, align 1
	volatile store i8 0, i8* @uc, align 1
	%14 = bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*> [#uses=1]
	volatile store i16 0, i16* %14, align 2
	%15 = bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*> [#uses=1]
	volatile store i16 0, i16* %15, align 2
	%16 = bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*> [#uses=1]
	volatile store i32 0, i32* %16, align 4
	%17 = bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*> [#uses=1]
	volatile store i32 0, i32* %17, align 4
	%18 = bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*> [#uses=1]
	volatile store i32 0, i32* %18, align 4
	%19 = bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*> [#uses=1]
	volatile store i32 0, i32* %19, align 4
	%20 = bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*> [#uses=1]
	volatile store i64 0, i64* %20, align 8
	%21 = bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*> [#uses=1]
	volatile store i64 0, i64* %21, align 8
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.swap.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.swap.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.swap.i32.p0i32(i32*, i32) nounwind

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind
