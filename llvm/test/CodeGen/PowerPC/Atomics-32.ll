; RUN: llvm-as < %s | llc -march=ppc32
; ModuleID = 'Atomics.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin9"
@sc = common global i8 0		; <i8*> [#uses=52]
@uc = common global i8 0		; <i8*> [#uses=100]
@ss = common global i16 0		; <i16*> [#uses=15]
@us = common global i16 0		; <i16*> [#uses=15]
@si = common global i32 0		; <i32*> [#uses=15]
@ui = common global i32 0		; <i32*> [#uses=23]
@sl = common global i32 0		; <i32*> [#uses=15]
@ul = common global i32 0		; <i32*> [#uses=15]
@sll = common global i64 0, align 8		; <i64*> [#uses=1]
@ull = common global i64 0, align 8		; <i64*> [#uses=1]

define void @test_op_ignore() nounwind {
entry:
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @sc, i8 1 )		; <i8>:0 [#uses=0]
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @uc, i8 1 )		; <i8>:1 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:2 [#uses=1]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %2, i16 1 )		; <i16>:3 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:4 [#uses=1]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %4, i16 1 )		; <i16>:5 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:6 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %6, i32 1 )		; <i32>:7 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:8 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %8, i32 1 )		; <i32>:9 [#uses=0]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:10 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %10, i32 1 )		; <i32>:11 [#uses=0]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:12 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %12, i32 1 )		; <i32>:13 [#uses=0]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @sc, i8 1 )		; <i8>:14 [#uses=0]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @uc, i8 1 )		; <i8>:15 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:16 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %16, i16 1 )		; <i16>:17 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:18 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %18, i16 1 )		; <i16>:19 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:20 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %20, i32 1 )		; <i32>:21 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:22 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %22, i32 1 )		; <i32>:23 [#uses=0]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:24 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %24, i32 1 )		; <i32>:25 [#uses=0]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:26 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %26, i32 1 )		; <i32>:27 [#uses=0]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @sc, i8 1 )		; <i8>:28 [#uses=0]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @uc, i8 1 )		; <i8>:29 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:30 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %30, i16 1 )		; <i16>:31 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:32 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %32, i16 1 )		; <i16>:33 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:34 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %34, i32 1 )		; <i32>:35 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:36 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %36, i32 1 )		; <i32>:37 [#uses=0]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:38 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %38, i32 1 )		; <i32>:39 [#uses=0]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:40 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %40, i32 1 )		; <i32>:41 [#uses=0]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @sc, i8 1 )		; <i8>:42 [#uses=0]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @uc, i8 1 )		; <i8>:43 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:44 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %44, i16 1 )		; <i16>:45 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:46 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %46, i16 1 )		; <i16>:47 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:48 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %48, i32 1 )		; <i32>:49 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:50 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %50, i32 1 )		; <i32>:51 [#uses=0]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:52 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %52, i32 1 )		; <i32>:53 [#uses=0]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:54 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %54, i32 1 )		; <i32>:55 [#uses=0]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @sc, i8 1 )		; <i8>:56 [#uses=0]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @uc, i8 1 )		; <i8>:57 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:58 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %58, i16 1 )		; <i16>:59 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:60 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %60, i16 1 )		; <i16>:61 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:62 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %62, i32 1 )		; <i32>:63 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:64 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %64, i32 1 )		; <i32>:65 [#uses=0]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:66 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %66, i32 1 )		; <i32>:67 [#uses=0]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:68 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %68, i32 1 )		; <i32>:69 [#uses=0]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @sc, i8 1 )		; <i8>:70 [#uses=0]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @uc, i8 1 )		; <i8>:71 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:72 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %72, i16 1 )		; <i16>:73 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:74 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %74, i16 1 )		; <i16>:75 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:76 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %76, i32 1 )		; <i32>:77 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:78 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %78, i32 1 )		; <i32>:79 [#uses=0]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:80 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %80, i32 1 )		; <i32>:81 [#uses=0]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:82 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %82, i32 1 )		; <i32>:83 [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.load.add.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.add.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.add.i32.p0i32(i32*, i32) nounwind

declare i8 @llvm.atomic.load.sub.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.sub.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.sub.i32.p0i32(i32*, i32) nounwind

declare i8 @llvm.atomic.load.or.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.or.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.or.i32.p0i32(i32*, i32) nounwind

declare i8 @llvm.atomic.load.xor.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.xor.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.xor.i32.p0i32(i32*, i32) nounwind

declare i8 @llvm.atomic.load.and.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.and.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.and.i32.p0i32(i32*, i32) nounwind

declare i8 @llvm.atomic.load.nand.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.load.nand.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.load.nand.i32.p0i32(i32*, i32) nounwind

define void @test_fetch_and_op() nounwind {
entry:
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @sc, i8 11 )		; <i8>:0 [#uses=1]
	store i8 %0, i8* @sc, align 1
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @uc, i8 11 )		; <i8>:1 [#uses=1]
	store i8 %1, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:2 [#uses=1]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %2, i16 11 )		; <i16>:3 [#uses=1]
	store i16 %3, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:4 [#uses=1]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %4, i16 11 )		; <i16>:5 [#uses=1]
	store i16 %5, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:6 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %6, i32 11 )		; <i32>:7 [#uses=1]
	store i32 %7, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:8 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %8, i32 11 )		; <i32>:9 [#uses=1]
	store i32 %9, i32* @ui, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:10 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %10, i32 11 )		; <i32>:11 [#uses=1]
	store i32 %11, i32* @sl, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:12 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %12, i32 11 )		; <i32>:13 [#uses=1]
	store i32 %13, i32* @ul, align 4
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @sc, i8 11 )		; <i8>:14 [#uses=1]
	store i8 %14, i8* @sc, align 1
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @uc, i8 11 )		; <i8>:15 [#uses=1]
	store i8 %15, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:16 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %16, i16 11 )		; <i16>:17 [#uses=1]
	store i16 %17, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:18 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %18, i16 11 )		; <i16>:19 [#uses=1]
	store i16 %19, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:20 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %20, i32 11 )		; <i32>:21 [#uses=1]
	store i32 %21, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:22 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %22, i32 11 )		; <i32>:23 [#uses=1]
	store i32 %23, i32* @ui, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:24 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %24, i32 11 )		; <i32>:25 [#uses=1]
	store i32 %25, i32* @sl, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:26 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %26, i32 11 )		; <i32>:27 [#uses=1]
	store i32 %27, i32* @ul, align 4
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @sc, i8 11 )		; <i8>:28 [#uses=1]
	store i8 %28, i8* @sc, align 1
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @uc, i8 11 )		; <i8>:29 [#uses=1]
	store i8 %29, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:30 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %30, i16 11 )		; <i16>:31 [#uses=1]
	store i16 %31, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:32 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %32, i16 11 )		; <i16>:33 [#uses=1]
	store i16 %33, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:34 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %34, i32 11 )		; <i32>:35 [#uses=1]
	store i32 %35, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:36 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %36, i32 11 )		; <i32>:37 [#uses=1]
	store i32 %37, i32* @ui, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:38 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %38, i32 11 )		; <i32>:39 [#uses=1]
	store i32 %39, i32* @sl, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:40 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %40, i32 11 )		; <i32>:41 [#uses=1]
	store i32 %41, i32* @ul, align 4
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @sc, i8 11 )		; <i8>:42 [#uses=1]
	store i8 %42, i8* @sc, align 1
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @uc, i8 11 )		; <i8>:43 [#uses=1]
	store i8 %43, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:44 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %44, i16 11 )		; <i16>:45 [#uses=1]
	store i16 %45, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:46 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %46, i16 11 )		; <i16>:47 [#uses=1]
	store i16 %47, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:48 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %48, i32 11 )		; <i32>:49 [#uses=1]
	store i32 %49, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:50 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %50, i32 11 )		; <i32>:51 [#uses=1]
	store i32 %51, i32* @ui, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:52 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %52, i32 11 )		; <i32>:53 [#uses=1]
	store i32 %53, i32* @sl, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:54 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %54, i32 11 )		; <i32>:55 [#uses=1]
	store i32 %55, i32* @ul, align 4
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @sc, i8 11 )		; <i8>:56 [#uses=1]
	store i8 %56, i8* @sc, align 1
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @uc, i8 11 )		; <i8>:57 [#uses=1]
	store i8 %57, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:58 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %58, i16 11 )		; <i16>:59 [#uses=1]
	store i16 %59, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:60 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %60, i16 11 )		; <i16>:61 [#uses=1]
	store i16 %61, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:62 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %62, i32 11 )		; <i32>:63 [#uses=1]
	store i32 %63, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:64 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %64, i32 11 )		; <i32>:65 [#uses=1]
	store i32 %65, i32* @ui, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:66 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %66, i32 11 )		; <i32>:67 [#uses=1]
	store i32 %67, i32* @sl, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:68 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %68, i32 11 )		; <i32>:69 [#uses=1]
	store i32 %69, i32* @ul, align 4
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @sc, i8 11 )		; <i8>:70 [#uses=1]
	store i8 %70, i8* @sc, align 1
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @uc, i8 11 )		; <i8>:71 [#uses=1]
	store i8 %71, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:72 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %72, i16 11 )		; <i16>:73 [#uses=1]
	store i16 %73, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:74 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %74, i16 11 )		; <i16>:75 [#uses=1]
	store i16 %75, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:76 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %76, i32 11 )		; <i32>:77 [#uses=1]
	store i32 %77, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:78 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %78, i32 11 )		; <i32>:79 [#uses=1]
	store i32 %79, i32* @ui, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:80 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %80, i32 11 )		; <i32>:81 [#uses=1]
	store i32 %81, i32* @sl, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:82 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %82, i32 11 )		; <i32>:83 [#uses=1]
	store i32 %83, i32* @ul, align 4
	br label %return

return:		; preds = %entry
	ret void
}

define void @test_op_and_fetch() nounwind {
entry:
	load i8* @uc, align 1		; <i8>:0 [#uses=2]
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @sc, i8 %0 )		; <i8>:1 [#uses=1]
	add i8 %1, %0		; <i8>:2 [#uses=1]
	store i8 %2, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:3 [#uses=2]
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @uc, i8 %3 )		; <i8>:4 [#uses=1]
	add i8 %4, %3		; <i8>:5 [#uses=1]
	store i8 %5, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:6 [#uses=1]
	zext i8 %6 to i16		; <i16>:7 [#uses=2]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:8 [#uses=1]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %8, i16 %7 )		; <i16>:9 [#uses=1]
	add i16 %9, %7		; <i16>:10 [#uses=1]
	store i16 %10, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:11 [#uses=1]
	zext i8 %11 to i16		; <i16>:12 [#uses=2]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:13 [#uses=1]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %13, i16 %12 )		; <i16>:14 [#uses=1]
	add i16 %14, %12		; <i16>:15 [#uses=1]
	store i16 %15, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:16 [#uses=1]
	zext i8 %16 to i32		; <i32>:17 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:18 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %18, i32 %17 )		; <i32>:19 [#uses=1]
	add i32 %19, %17		; <i32>:20 [#uses=1]
	store i32 %20, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:21 [#uses=1]
	zext i8 %21 to i32		; <i32>:22 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:23 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %23, i32 %22 )		; <i32>:24 [#uses=1]
	add i32 %24, %22		; <i32>:25 [#uses=1]
	store i32 %25, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:26 [#uses=1]
	zext i8 %26 to i32		; <i32>:27 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:28 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %28, i32 %27 )		; <i32>:29 [#uses=1]
	add i32 %29, %27		; <i32>:30 [#uses=1]
	store i32 %30, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:31 [#uses=1]
	zext i8 %31 to i32		; <i32>:32 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:33 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %33, i32 %32 )		; <i32>:34 [#uses=1]
	add i32 %34, %32		; <i32>:35 [#uses=1]
	store i32 %35, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:36 [#uses=2]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @sc, i8 %36 )		; <i8>:37 [#uses=1]
	sub i8 %37, %36		; <i8>:38 [#uses=1]
	store i8 %38, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:39 [#uses=2]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @uc, i8 %39 )		; <i8>:40 [#uses=1]
	sub i8 %40, %39		; <i8>:41 [#uses=1]
	store i8 %41, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:42 [#uses=1]
	zext i8 %42 to i16		; <i16>:43 [#uses=2]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:44 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %44, i16 %43 )		; <i16>:45 [#uses=1]
	sub i16 %45, %43		; <i16>:46 [#uses=1]
	store i16 %46, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:47 [#uses=1]
	zext i8 %47 to i16		; <i16>:48 [#uses=2]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:49 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %49, i16 %48 )		; <i16>:50 [#uses=1]
	sub i16 %50, %48		; <i16>:51 [#uses=1]
	store i16 %51, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:52 [#uses=1]
	zext i8 %52 to i32		; <i32>:53 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:54 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %54, i32 %53 )		; <i32>:55 [#uses=1]
	sub i32 %55, %53		; <i32>:56 [#uses=1]
	store i32 %56, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:57 [#uses=1]
	zext i8 %57 to i32		; <i32>:58 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:59 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %59, i32 %58 )		; <i32>:60 [#uses=1]
	sub i32 %60, %58		; <i32>:61 [#uses=1]
	store i32 %61, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:62 [#uses=1]
	zext i8 %62 to i32		; <i32>:63 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:64 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %64, i32 %63 )		; <i32>:65 [#uses=1]
	sub i32 %65, %63		; <i32>:66 [#uses=1]
	store i32 %66, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:67 [#uses=1]
	zext i8 %67 to i32		; <i32>:68 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:69 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %69, i32 %68 )		; <i32>:70 [#uses=1]
	sub i32 %70, %68		; <i32>:71 [#uses=1]
	store i32 %71, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:72 [#uses=2]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @sc, i8 %72 )		; <i8>:73 [#uses=1]
	or i8 %73, %72		; <i8>:74 [#uses=1]
	store i8 %74, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:75 [#uses=2]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @uc, i8 %75 )		; <i8>:76 [#uses=1]
	or i8 %76, %75		; <i8>:77 [#uses=1]
	store i8 %77, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:78 [#uses=1]
	zext i8 %78 to i16		; <i16>:79 [#uses=2]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:80 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %80, i16 %79 )		; <i16>:81 [#uses=1]
	or i16 %81, %79		; <i16>:82 [#uses=1]
	store i16 %82, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:83 [#uses=1]
	zext i8 %83 to i16		; <i16>:84 [#uses=2]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:85 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %85, i16 %84 )		; <i16>:86 [#uses=1]
	or i16 %86, %84		; <i16>:87 [#uses=1]
	store i16 %87, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:88 [#uses=1]
	zext i8 %88 to i32		; <i32>:89 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:90 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %90, i32 %89 )		; <i32>:91 [#uses=1]
	or i32 %91, %89		; <i32>:92 [#uses=1]
	store i32 %92, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:93 [#uses=1]
	zext i8 %93 to i32		; <i32>:94 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:95 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %95, i32 %94 )		; <i32>:96 [#uses=1]
	or i32 %96, %94		; <i32>:97 [#uses=1]
	store i32 %97, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:98 [#uses=1]
	zext i8 %98 to i32		; <i32>:99 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:100 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %100, i32 %99 )		; <i32>:101 [#uses=1]
	or i32 %101, %99		; <i32>:102 [#uses=1]
	store i32 %102, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:103 [#uses=1]
	zext i8 %103 to i32		; <i32>:104 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:105 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %105, i32 %104 )		; <i32>:106 [#uses=1]
	or i32 %106, %104		; <i32>:107 [#uses=1]
	store i32 %107, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:108 [#uses=2]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @sc, i8 %108 )		; <i8>:109 [#uses=1]
	xor i8 %109, %108		; <i8>:110 [#uses=1]
	store i8 %110, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:111 [#uses=2]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @uc, i8 %111 )		; <i8>:112 [#uses=1]
	xor i8 %112, %111		; <i8>:113 [#uses=1]
	store i8 %113, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:114 [#uses=1]
	zext i8 %114 to i16		; <i16>:115 [#uses=2]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:116 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %116, i16 %115 )		; <i16>:117 [#uses=1]
	xor i16 %117, %115		; <i16>:118 [#uses=1]
	store i16 %118, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:119 [#uses=1]
	zext i8 %119 to i16		; <i16>:120 [#uses=2]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:121 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %121, i16 %120 )		; <i16>:122 [#uses=1]
	xor i16 %122, %120		; <i16>:123 [#uses=1]
	store i16 %123, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:124 [#uses=1]
	zext i8 %124 to i32		; <i32>:125 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:126 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %126, i32 %125 )		; <i32>:127 [#uses=1]
	xor i32 %127, %125		; <i32>:128 [#uses=1]
	store i32 %128, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:129 [#uses=1]
	zext i8 %129 to i32		; <i32>:130 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:131 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %131, i32 %130 )		; <i32>:132 [#uses=1]
	xor i32 %132, %130		; <i32>:133 [#uses=1]
	store i32 %133, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:134 [#uses=1]
	zext i8 %134 to i32		; <i32>:135 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:136 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %136, i32 %135 )		; <i32>:137 [#uses=1]
	xor i32 %137, %135		; <i32>:138 [#uses=1]
	store i32 %138, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:139 [#uses=1]
	zext i8 %139 to i32		; <i32>:140 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:141 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %141, i32 %140 )		; <i32>:142 [#uses=1]
	xor i32 %142, %140		; <i32>:143 [#uses=1]
	store i32 %143, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:144 [#uses=2]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @sc, i8 %144 )		; <i8>:145 [#uses=1]
	and i8 %145, %144		; <i8>:146 [#uses=1]
	store i8 %146, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:147 [#uses=2]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @uc, i8 %147 )		; <i8>:148 [#uses=1]
	and i8 %148, %147		; <i8>:149 [#uses=1]
	store i8 %149, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:150 [#uses=1]
	zext i8 %150 to i16		; <i16>:151 [#uses=2]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:152 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %152, i16 %151 )		; <i16>:153 [#uses=1]
	and i16 %153, %151		; <i16>:154 [#uses=1]
	store i16 %154, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:155 [#uses=1]
	zext i8 %155 to i16		; <i16>:156 [#uses=2]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:157 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %157, i16 %156 )		; <i16>:158 [#uses=1]
	and i16 %158, %156		; <i16>:159 [#uses=1]
	store i16 %159, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:160 [#uses=1]
	zext i8 %160 to i32		; <i32>:161 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:162 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %162, i32 %161 )		; <i32>:163 [#uses=1]
	and i32 %163, %161		; <i32>:164 [#uses=1]
	store i32 %164, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:165 [#uses=1]
	zext i8 %165 to i32		; <i32>:166 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:167 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %167, i32 %166 )		; <i32>:168 [#uses=1]
	and i32 %168, %166		; <i32>:169 [#uses=1]
	store i32 %169, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:170 [#uses=1]
	zext i8 %170 to i32		; <i32>:171 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:172 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %172, i32 %171 )		; <i32>:173 [#uses=1]
	and i32 %173, %171		; <i32>:174 [#uses=1]
	store i32 %174, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:175 [#uses=1]
	zext i8 %175 to i32		; <i32>:176 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:177 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %177, i32 %176 )		; <i32>:178 [#uses=1]
	and i32 %178, %176		; <i32>:179 [#uses=1]
	store i32 %179, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:180 [#uses=2]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @sc, i8 %180 )		; <i8>:181 [#uses=1]
	xor i8 %181, -1		; <i8>:182 [#uses=1]
	and i8 %182, %180		; <i8>:183 [#uses=1]
	store i8 %183, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:184 [#uses=2]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @uc, i8 %184 )		; <i8>:185 [#uses=1]
	xor i8 %185, -1		; <i8>:186 [#uses=1]
	and i8 %186, %184		; <i8>:187 [#uses=1]
	store i8 %187, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:188 [#uses=1]
	zext i8 %188 to i16		; <i16>:189 [#uses=2]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:190 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %190, i16 %189 )		; <i16>:191 [#uses=1]
	xor i16 %191, -1		; <i16>:192 [#uses=1]
	and i16 %192, %189		; <i16>:193 [#uses=1]
	store i16 %193, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:194 [#uses=1]
	zext i8 %194 to i16		; <i16>:195 [#uses=2]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:196 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %196, i16 %195 )		; <i16>:197 [#uses=1]
	xor i16 %197, -1		; <i16>:198 [#uses=1]
	and i16 %198, %195		; <i16>:199 [#uses=1]
	store i16 %199, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:200 [#uses=1]
	zext i8 %200 to i32		; <i32>:201 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:202 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %202, i32 %201 )		; <i32>:203 [#uses=1]
	xor i32 %203, -1		; <i32>:204 [#uses=1]
	and i32 %204, %201		; <i32>:205 [#uses=1]
	store i32 %205, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:206 [#uses=1]
	zext i8 %206 to i32		; <i32>:207 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:208 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %208, i32 %207 )		; <i32>:209 [#uses=1]
	xor i32 %209, -1		; <i32>:210 [#uses=1]
	and i32 %210, %207		; <i32>:211 [#uses=1]
	store i32 %211, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:212 [#uses=1]
	zext i8 %212 to i32		; <i32>:213 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:214 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %214, i32 %213 )		; <i32>:215 [#uses=1]
	xor i32 %215, -1		; <i32>:216 [#uses=1]
	and i32 %216, %213		; <i32>:217 [#uses=1]
	store i32 %217, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:218 [#uses=1]
	zext i8 %218 to i32		; <i32>:219 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:220 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %220, i32 %219 )		; <i32>:221 [#uses=1]
	xor i32 %221, -1		; <i32>:222 [#uses=1]
	and i32 %222, %219		; <i32>:223 [#uses=1]
	store i32 %223, i32* @ul, align 4
	br label %return

return:		; preds = %entry
	ret void
}

define void @test_compare_and_swap() nounwind {
entry:
	load i8* @uc, align 1		; <i8>:0 [#uses=1]
	load i8* @sc, align 1		; <i8>:1 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @sc, i8 %0, i8 %1 )		; <i8>:2 [#uses=1]
	store i8 %2, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:3 [#uses=1]
	load i8* @sc, align 1		; <i8>:4 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @uc, i8 %3, i8 %4 )		; <i8>:5 [#uses=1]
	store i8 %5, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:6 [#uses=1]
	zext i8 %6 to i16		; <i16>:7 [#uses=1]
	load i8* @sc, align 1		; <i8>:8 [#uses=1]
	sext i8 %8 to i16		; <i16>:9 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:10 [#uses=1]
	call i16 @llvm.atomic.cmp.swap.i16.p0i16( i16* %10, i16 %7, i16 %9 )		; <i16>:11 [#uses=1]
	store i16 %11, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:12 [#uses=1]
	zext i8 %12 to i16		; <i16>:13 [#uses=1]
	load i8* @sc, align 1		; <i8>:14 [#uses=1]
	sext i8 %14 to i16		; <i16>:15 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:16 [#uses=1]
	call i16 @llvm.atomic.cmp.swap.i16.p0i16( i16* %16, i16 %13, i16 %15 )		; <i16>:17 [#uses=1]
	store i16 %17, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:18 [#uses=1]
	zext i8 %18 to i32		; <i32>:19 [#uses=1]
	load i8* @sc, align 1		; <i8>:20 [#uses=1]
	sext i8 %20 to i32		; <i32>:21 [#uses=1]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:22 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %22, i32 %19, i32 %21 )		; <i32>:23 [#uses=1]
	store i32 %23, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:24 [#uses=1]
	zext i8 %24 to i32		; <i32>:25 [#uses=1]
	load i8* @sc, align 1		; <i8>:26 [#uses=1]
	sext i8 %26 to i32		; <i32>:27 [#uses=1]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:28 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %28, i32 %25, i32 %27 )		; <i32>:29 [#uses=1]
	store i32 %29, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:30 [#uses=1]
	zext i8 %30 to i32		; <i32>:31 [#uses=1]
	load i8* @sc, align 1		; <i8>:32 [#uses=1]
	sext i8 %32 to i32		; <i32>:33 [#uses=1]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:34 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %34, i32 %31, i32 %33 )		; <i32>:35 [#uses=1]
	store i32 %35, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:36 [#uses=1]
	zext i8 %36 to i32		; <i32>:37 [#uses=1]
	load i8* @sc, align 1		; <i8>:38 [#uses=1]
	sext i8 %38 to i32		; <i32>:39 [#uses=1]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:40 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %40, i32 %37, i32 %39 )		; <i32>:41 [#uses=1]
	store i32 %41, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:42 [#uses=2]
	load i8* @sc, align 1		; <i8>:43 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @sc, i8 %42, i8 %43 )		; <i8>:44 [#uses=1]
	icmp eq i8 %44, %42		; <i1>:45 [#uses=1]
	zext i1 %45 to i32		; <i32>:46 [#uses=1]
	store i32 %46, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:47 [#uses=2]
	load i8* @sc, align 1		; <i8>:48 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @uc, i8 %47, i8 %48 )		; <i8>:49 [#uses=1]
	icmp eq i8 %49, %47		; <i1>:50 [#uses=1]
	zext i1 %50 to i32		; <i32>:51 [#uses=1]
	store i32 %51, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:52 [#uses=1]
	zext i8 %52 to i16		; <i16>:53 [#uses=2]
	load i8* @sc, align 1		; <i8>:54 [#uses=1]
	sext i8 %54 to i16		; <i16>:55 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:56 [#uses=1]
	call i16 @llvm.atomic.cmp.swap.i16.p0i16( i16* %56, i16 %53, i16 %55 )		; <i16>:57 [#uses=1]
	icmp eq i16 %57, %53		; <i1>:58 [#uses=1]
	zext i1 %58 to i32		; <i32>:59 [#uses=1]
	store i32 %59, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:60 [#uses=1]
	zext i8 %60 to i16		; <i16>:61 [#uses=2]
	load i8* @sc, align 1		; <i8>:62 [#uses=1]
	sext i8 %62 to i16		; <i16>:63 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:64 [#uses=1]
	call i16 @llvm.atomic.cmp.swap.i16.p0i16( i16* %64, i16 %61, i16 %63 )		; <i16>:65 [#uses=1]
	icmp eq i16 %65, %61		; <i1>:66 [#uses=1]
	zext i1 %66 to i32		; <i32>:67 [#uses=1]
	store i32 %67, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:68 [#uses=1]
	zext i8 %68 to i32		; <i32>:69 [#uses=2]
	load i8* @sc, align 1		; <i8>:70 [#uses=1]
	sext i8 %70 to i32		; <i32>:71 [#uses=1]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:72 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %72, i32 %69, i32 %71 )		; <i32>:73 [#uses=1]
	icmp eq i32 %73, %69		; <i1>:74 [#uses=1]
	zext i1 %74 to i32		; <i32>:75 [#uses=1]
	store i32 %75, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:76 [#uses=1]
	zext i8 %76 to i32		; <i32>:77 [#uses=2]
	load i8* @sc, align 1		; <i8>:78 [#uses=1]
	sext i8 %78 to i32		; <i32>:79 [#uses=1]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:80 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %80, i32 %77, i32 %79 )		; <i32>:81 [#uses=1]
	icmp eq i32 %81, %77		; <i1>:82 [#uses=1]
	zext i1 %82 to i32		; <i32>:83 [#uses=1]
	store i32 %83, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:84 [#uses=1]
	zext i8 %84 to i32		; <i32>:85 [#uses=2]
	load i8* @sc, align 1		; <i8>:86 [#uses=1]
	sext i8 %86 to i32		; <i32>:87 [#uses=1]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:88 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %88, i32 %85, i32 %87 )		; <i32>:89 [#uses=1]
	icmp eq i32 %89, %85		; <i1>:90 [#uses=1]
	zext i1 %90 to i32		; <i32>:91 [#uses=1]
	store i32 %91, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:92 [#uses=1]
	zext i8 %92 to i32		; <i32>:93 [#uses=2]
	load i8* @sc, align 1		; <i8>:94 [#uses=1]
	sext i8 %94 to i32		; <i32>:95 [#uses=1]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:96 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %96, i32 %93, i32 %95 )		; <i32>:97 [#uses=1]
	icmp eq i32 %97, %93		; <i1>:98 [#uses=1]
	zext i1 %98 to i32		; <i32>:99 [#uses=1]
	store i32 %99, i32* @ui, align 4
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.cmp.swap.i8.p0i8(i8*, i8, i8) nounwind

declare i16 @llvm.atomic.cmp.swap.i16.p0i16(i16*, i16, i16) nounwind

declare i32 @llvm.atomic.cmp.swap.i32.p0i32(i32*, i32, i32) nounwind

define void @test_lock() nounwind {
entry:
	call i8 @llvm.atomic.swap.i8.p0i8( i8* @sc, i8 1 )		; <i8>:0 [#uses=1]
	store i8 %0, i8* @sc, align 1
	call i8 @llvm.atomic.swap.i8.p0i8( i8* @uc, i8 1 )		; <i8>:1 [#uses=1]
	store i8 %1, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:2 [#uses=1]
	call i16 @llvm.atomic.swap.i16.p0i16( i16* %2, i16 1 )		; <i16>:3 [#uses=1]
	store i16 %3, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:4 [#uses=1]
	call i16 @llvm.atomic.swap.i16.p0i16( i16* %4, i16 1 )		; <i16>:5 [#uses=1]
	store i16 %5, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:6 [#uses=1]
	call i32 @llvm.atomic.swap.i32.p0i32( i32* %6, i32 1 )		; <i32>:7 [#uses=1]
	store i32 %7, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:8 [#uses=1]
	call i32 @llvm.atomic.swap.i32.p0i32( i32* %8, i32 1 )		; <i32>:9 [#uses=1]
	store i32 %9, i32* @ui, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:10 [#uses=1]
	call i32 @llvm.atomic.swap.i32.p0i32( i32* %10, i32 1 )		; <i32>:11 [#uses=1]
	store i32 %11, i32* @sl, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:12 [#uses=1]
	call i32 @llvm.atomic.swap.i32.p0i32( i32* %12, i32 1 )		; <i32>:13 [#uses=1]
	store i32 %13, i32* @ul, align 4
	call void @llvm.memory.barrier( i1 true, i1 true, i1 true, i1 true, i1 false )
	volatile store i8 0, i8* @sc, align 1
	volatile store i8 0, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:14 [#uses=1]
	volatile store i16 0, i16* %14, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:15 [#uses=1]
	volatile store i16 0, i16* %15, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:16 [#uses=1]
	volatile store i32 0, i32* %16, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:17 [#uses=1]
	volatile store i32 0, i32* %17, align 4
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:18 [#uses=1]
	volatile store i32 0, i32* %18, align 4
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:19 [#uses=1]
	volatile store i32 0, i32* %19, align 4
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:20 [#uses=1]
	volatile store i64 0, i64* %20, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:21 [#uses=1]
	volatile store i64 0, i64* %21, align 8
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.swap.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.swap.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.swap.i32.p0i32(i32*, i32) nounwind

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind
