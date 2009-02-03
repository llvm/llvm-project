; RUN: llvm-as < %s | llc -march=x86
;; Note the 64-bit variants are not supported yet (in 32-bit mode).
; ModuleID = 'Atomics.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@sc = common global i8 0		; <i8*> [#uses=52]
@uc = common global i8 0		; <i8*> [#uses=100]
@ss = common global i16 0		; <i16*> [#uses=15]
@us = common global i16 0		; <i16*> [#uses=15]
@si = common global i32 0		; <i32*> [#uses=15]
@ui = common global i32 0		; <i32*> [#uses=23]
@sl = common global i32 0		; <i32*> [#uses=15]
@ul = common global i32 0		; <i32*> [#uses=15]

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
	load i8* @uc, align 1		; <i8>:0 [#uses=1]
	zext i8 %0 to i32		; <i32>:1 [#uses=1]
	trunc i32 %1 to i8		; <i8>:2 [#uses=2]
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @sc, i8 %2 )		; <i8>:3 [#uses=1]
	add i8 %3, %2		; <i8>:4 [#uses=1]
	store i8 %4, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:5 [#uses=1]
	zext i8 %5 to i32		; <i32>:6 [#uses=1]
	trunc i32 %6 to i8		; <i8>:7 [#uses=2]
	call i8 @llvm.atomic.load.add.i8.p0i8( i8* @uc, i8 %7 )		; <i8>:8 [#uses=1]
	add i8 %8, %7		; <i8>:9 [#uses=1]
	store i8 %9, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:10 [#uses=1]
	zext i8 %10 to i32		; <i32>:11 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:12 [#uses=1]
	trunc i32 %11 to i16		; <i16>:13 [#uses=2]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %12, i16 %13 )		; <i16>:14 [#uses=1]
	add i16 %14, %13		; <i16>:15 [#uses=1]
	store i16 %15, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:16 [#uses=1]
	zext i8 %16 to i32		; <i32>:17 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:18 [#uses=1]
	trunc i32 %17 to i16		; <i16>:19 [#uses=2]
	call i16 @llvm.atomic.load.add.i16.p0i16( i16* %18, i16 %19 )		; <i16>:20 [#uses=1]
	add i16 %20, %19		; <i16>:21 [#uses=1]
	store i16 %21, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:22 [#uses=1]
	zext i8 %22 to i32		; <i32>:23 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:24 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %24, i32 %23 )		; <i32>:25 [#uses=1]
	add i32 %25, %23		; <i32>:26 [#uses=1]
	store i32 %26, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:27 [#uses=1]
	zext i8 %27 to i32		; <i32>:28 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:29 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %29, i32 %28 )		; <i32>:30 [#uses=1]
	add i32 %30, %28		; <i32>:31 [#uses=1]
	store i32 %31, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:32 [#uses=1]
	zext i8 %32 to i32		; <i32>:33 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:34 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %34, i32 %33 )		; <i32>:35 [#uses=1]
	add i32 %35, %33		; <i32>:36 [#uses=1]
	store i32 %36, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:37 [#uses=1]
	zext i8 %37 to i32		; <i32>:38 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:39 [#uses=1]
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %39, i32 %38 )		; <i32>:40 [#uses=1]
	add i32 %40, %38		; <i32>:41 [#uses=1]
	store i32 %41, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:42 [#uses=1]
	zext i8 %42 to i32		; <i32>:43 [#uses=1]
	trunc i32 %43 to i8		; <i8>:44 [#uses=2]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @sc, i8 %44 )		; <i8>:45 [#uses=1]
	sub i8 %45, %44		; <i8>:46 [#uses=1]
	store i8 %46, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:47 [#uses=1]
	zext i8 %47 to i32		; <i32>:48 [#uses=1]
	trunc i32 %48 to i8		; <i8>:49 [#uses=2]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @uc, i8 %49 )		; <i8>:50 [#uses=1]
	sub i8 %50, %49		; <i8>:51 [#uses=1]
	store i8 %51, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:52 [#uses=1]
	zext i8 %52 to i32		; <i32>:53 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:54 [#uses=1]
	trunc i32 %53 to i16		; <i16>:55 [#uses=2]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %54, i16 %55 )		; <i16>:56 [#uses=1]
	sub i16 %56, %55		; <i16>:57 [#uses=1]
	store i16 %57, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:58 [#uses=1]
	zext i8 %58 to i32		; <i32>:59 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:60 [#uses=1]
	trunc i32 %59 to i16		; <i16>:61 [#uses=2]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %60, i16 %61 )		; <i16>:62 [#uses=1]
	sub i16 %62, %61		; <i16>:63 [#uses=1]
	store i16 %63, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:64 [#uses=1]
	zext i8 %64 to i32		; <i32>:65 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:66 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %66, i32 %65 )		; <i32>:67 [#uses=1]
	sub i32 %67, %65		; <i32>:68 [#uses=1]
	store i32 %68, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:69 [#uses=1]
	zext i8 %69 to i32		; <i32>:70 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:71 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %71, i32 %70 )		; <i32>:72 [#uses=1]
	sub i32 %72, %70		; <i32>:73 [#uses=1]
	store i32 %73, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:74 [#uses=1]
	zext i8 %74 to i32		; <i32>:75 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:76 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %76, i32 %75 )		; <i32>:77 [#uses=1]
	sub i32 %77, %75		; <i32>:78 [#uses=1]
	store i32 %78, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:79 [#uses=1]
	zext i8 %79 to i32		; <i32>:80 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:81 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %81, i32 %80 )		; <i32>:82 [#uses=1]
	sub i32 %82, %80		; <i32>:83 [#uses=1]
	store i32 %83, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:84 [#uses=1]
	zext i8 %84 to i32		; <i32>:85 [#uses=1]
	trunc i32 %85 to i8		; <i8>:86 [#uses=2]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @sc, i8 %86 )		; <i8>:87 [#uses=1]
	or i8 %87, %86		; <i8>:88 [#uses=1]
	store i8 %88, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:89 [#uses=1]
	zext i8 %89 to i32		; <i32>:90 [#uses=1]
	trunc i32 %90 to i8		; <i8>:91 [#uses=2]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @uc, i8 %91 )		; <i8>:92 [#uses=1]
	or i8 %92, %91		; <i8>:93 [#uses=1]
	store i8 %93, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:94 [#uses=1]
	zext i8 %94 to i32		; <i32>:95 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:96 [#uses=1]
	trunc i32 %95 to i16		; <i16>:97 [#uses=2]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %96, i16 %97 )		; <i16>:98 [#uses=1]
	or i16 %98, %97		; <i16>:99 [#uses=1]
	store i16 %99, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:100 [#uses=1]
	zext i8 %100 to i32		; <i32>:101 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:102 [#uses=1]
	trunc i32 %101 to i16		; <i16>:103 [#uses=2]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %102, i16 %103 )		; <i16>:104 [#uses=1]
	or i16 %104, %103		; <i16>:105 [#uses=1]
	store i16 %105, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:106 [#uses=1]
	zext i8 %106 to i32		; <i32>:107 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:108 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %108, i32 %107 )		; <i32>:109 [#uses=1]
	or i32 %109, %107		; <i32>:110 [#uses=1]
	store i32 %110, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:111 [#uses=1]
	zext i8 %111 to i32		; <i32>:112 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:113 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %113, i32 %112 )		; <i32>:114 [#uses=1]
	or i32 %114, %112		; <i32>:115 [#uses=1]
	store i32 %115, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:116 [#uses=1]
	zext i8 %116 to i32		; <i32>:117 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:118 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %118, i32 %117 )		; <i32>:119 [#uses=1]
	or i32 %119, %117		; <i32>:120 [#uses=1]
	store i32 %120, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:121 [#uses=1]
	zext i8 %121 to i32		; <i32>:122 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:123 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %123, i32 %122 )		; <i32>:124 [#uses=1]
	or i32 %124, %122		; <i32>:125 [#uses=1]
	store i32 %125, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:126 [#uses=1]
	zext i8 %126 to i32		; <i32>:127 [#uses=1]
	trunc i32 %127 to i8		; <i8>:128 [#uses=2]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @sc, i8 %128 )		; <i8>:129 [#uses=1]
	xor i8 %129, %128		; <i8>:130 [#uses=1]
	store i8 %130, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:131 [#uses=1]
	zext i8 %131 to i32		; <i32>:132 [#uses=1]
	trunc i32 %132 to i8		; <i8>:133 [#uses=2]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @uc, i8 %133 )		; <i8>:134 [#uses=1]
	xor i8 %134, %133		; <i8>:135 [#uses=1]
	store i8 %135, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:136 [#uses=1]
	zext i8 %136 to i32		; <i32>:137 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:138 [#uses=1]
	trunc i32 %137 to i16		; <i16>:139 [#uses=2]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %138, i16 %139 )		; <i16>:140 [#uses=1]
	xor i16 %140, %139		; <i16>:141 [#uses=1]
	store i16 %141, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:142 [#uses=1]
	zext i8 %142 to i32		; <i32>:143 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:144 [#uses=1]
	trunc i32 %143 to i16		; <i16>:145 [#uses=2]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %144, i16 %145 )		; <i16>:146 [#uses=1]
	xor i16 %146, %145		; <i16>:147 [#uses=1]
	store i16 %147, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:148 [#uses=1]
	zext i8 %148 to i32		; <i32>:149 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:150 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %150, i32 %149 )		; <i32>:151 [#uses=1]
	xor i32 %151, %149		; <i32>:152 [#uses=1]
	store i32 %152, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:153 [#uses=1]
	zext i8 %153 to i32		; <i32>:154 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:155 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %155, i32 %154 )		; <i32>:156 [#uses=1]
	xor i32 %156, %154		; <i32>:157 [#uses=1]
	store i32 %157, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:158 [#uses=1]
	zext i8 %158 to i32		; <i32>:159 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:160 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %160, i32 %159 )		; <i32>:161 [#uses=1]
	xor i32 %161, %159		; <i32>:162 [#uses=1]
	store i32 %162, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:163 [#uses=1]
	zext i8 %163 to i32		; <i32>:164 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:165 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %165, i32 %164 )		; <i32>:166 [#uses=1]
	xor i32 %166, %164		; <i32>:167 [#uses=1]
	store i32 %167, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:168 [#uses=1]
	zext i8 %168 to i32		; <i32>:169 [#uses=1]
	trunc i32 %169 to i8		; <i8>:170 [#uses=2]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @sc, i8 %170 )		; <i8>:171 [#uses=1]
	and i8 %171, %170		; <i8>:172 [#uses=1]
	store i8 %172, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:173 [#uses=1]
	zext i8 %173 to i32		; <i32>:174 [#uses=1]
	trunc i32 %174 to i8		; <i8>:175 [#uses=2]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @uc, i8 %175 )		; <i8>:176 [#uses=1]
	and i8 %176, %175		; <i8>:177 [#uses=1]
	store i8 %177, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:178 [#uses=1]
	zext i8 %178 to i32		; <i32>:179 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:180 [#uses=1]
	trunc i32 %179 to i16		; <i16>:181 [#uses=2]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %180, i16 %181 )		; <i16>:182 [#uses=1]
	and i16 %182, %181		; <i16>:183 [#uses=1]
	store i16 %183, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:184 [#uses=1]
	zext i8 %184 to i32		; <i32>:185 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:186 [#uses=1]
	trunc i32 %185 to i16		; <i16>:187 [#uses=2]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %186, i16 %187 )		; <i16>:188 [#uses=1]
	and i16 %188, %187		; <i16>:189 [#uses=1]
	store i16 %189, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:190 [#uses=1]
	zext i8 %190 to i32		; <i32>:191 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:192 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %192, i32 %191 )		; <i32>:193 [#uses=1]
	and i32 %193, %191		; <i32>:194 [#uses=1]
	store i32 %194, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:195 [#uses=1]
	zext i8 %195 to i32		; <i32>:196 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:197 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %197, i32 %196 )		; <i32>:198 [#uses=1]
	and i32 %198, %196		; <i32>:199 [#uses=1]
	store i32 %199, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:200 [#uses=1]
	zext i8 %200 to i32		; <i32>:201 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:202 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %202, i32 %201 )		; <i32>:203 [#uses=1]
	and i32 %203, %201		; <i32>:204 [#uses=1]
	store i32 %204, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:205 [#uses=1]
	zext i8 %205 to i32		; <i32>:206 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:207 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %207, i32 %206 )		; <i32>:208 [#uses=1]
	and i32 %208, %206		; <i32>:209 [#uses=1]
	store i32 %209, i32* @ul, align 4
	load i8* @uc, align 1		; <i8>:210 [#uses=1]
	zext i8 %210 to i32		; <i32>:211 [#uses=1]
	trunc i32 %211 to i8		; <i8>:212 [#uses=2]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @sc, i8 %212 )		; <i8>:213 [#uses=1]
	xor i8 %213, -1		; <i8>:214 [#uses=1]
	and i8 %214, %212		; <i8>:215 [#uses=1]
	store i8 %215, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:216 [#uses=1]
	zext i8 %216 to i32		; <i32>:217 [#uses=1]
	trunc i32 %217 to i8		; <i8>:218 [#uses=2]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @uc, i8 %218 )		; <i8>:219 [#uses=1]
	xor i8 %219, -1		; <i8>:220 [#uses=1]
	and i8 %220, %218		; <i8>:221 [#uses=1]
	store i8 %221, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:222 [#uses=1]
	zext i8 %222 to i32		; <i32>:223 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:224 [#uses=1]
	trunc i32 %223 to i16		; <i16>:225 [#uses=2]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %224, i16 %225 )		; <i16>:226 [#uses=1]
	xor i16 %226, -1		; <i16>:227 [#uses=1]
	and i16 %227, %225		; <i16>:228 [#uses=1]
	store i16 %228, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:229 [#uses=1]
	zext i8 %229 to i32		; <i32>:230 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:231 [#uses=1]
	trunc i32 %230 to i16		; <i16>:232 [#uses=2]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %231, i16 %232 )		; <i16>:233 [#uses=1]
	xor i16 %233, -1		; <i16>:234 [#uses=1]
	and i16 %234, %232		; <i16>:235 [#uses=1]
	store i16 %235, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:236 [#uses=1]
	zext i8 %236 to i32		; <i32>:237 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:238 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %238, i32 %237 )		; <i32>:239 [#uses=1]
	xor i32 %239, -1		; <i32>:240 [#uses=1]
	and i32 %240, %237		; <i32>:241 [#uses=1]
	store i32 %241, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:242 [#uses=1]
	zext i8 %242 to i32		; <i32>:243 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:244 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %244, i32 %243 )		; <i32>:245 [#uses=1]
	xor i32 %245, -1		; <i32>:246 [#uses=1]
	and i32 %246, %243		; <i32>:247 [#uses=1]
	store i32 %247, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:248 [#uses=1]
	zext i8 %248 to i32		; <i32>:249 [#uses=2]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:250 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %250, i32 %249 )		; <i32>:251 [#uses=1]
	xor i32 %251, -1		; <i32>:252 [#uses=1]
	and i32 %252, %249		; <i32>:253 [#uses=1]
	store i32 %253, i32* @sl, align 4
	load i8* @uc, align 1		; <i8>:254 [#uses=1]
	zext i8 %254 to i32		; <i32>:255 [#uses=2]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:256 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %256, i32 %255 )		; <i32>:257 [#uses=1]
	xor i32 %257, -1		; <i32>:258 [#uses=1]
	and i32 %258, %255		; <i32>:259 [#uses=1]
	store i32 %259, i32* @ul, align 4
	br label %return

return:		; preds = %entry
	ret void
}

define void @test_compare_and_swap() nounwind {
entry:
	load i8* @sc, align 1		; <i8>:0 [#uses=1]
	zext i8 %0 to i32		; <i32>:1 [#uses=1]
	load i8* @uc, align 1		; <i8>:2 [#uses=1]
	zext i8 %2 to i32		; <i32>:3 [#uses=1]
	trunc i32 %3 to i8		; <i8>:4 [#uses=1]
	trunc i32 %1 to i8		; <i8>:5 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @sc, i8 %4, i8 %5 )		; <i8>:6 [#uses=1]
	store i8 %6, i8* @sc, align 1
	load i8* @sc, align 1		; <i8>:7 [#uses=1]
	zext i8 %7 to i32		; <i32>:8 [#uses=1]
	load i8* @uc, align 1		; <i8>:9 [#uses=1]
	zext i8 %9 to i32		; <i32>:10 [#uses=1]
	trunc i32 %10 to i8		; <i8>:11 [#uses=1]
	trunc i32 %8 to i8		; <i8>:12 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @uc, i8 %11, i8 %12 )		; <i8>:13 [#uses=1]
	store i8 %13, i8* @uc, align 1
	load i8* @sc, align 1		; <i8>:14 [#uses=1]
	sext i8 %14 to i16		; <i16>:15 [#uses=1]
	zext i16 %15 to i32		; <i32>:16 [#uses=1]
	load i8* @uc, align 1		; <i8>:17 [#uses=1]
	zext i8 %17 to i32		; <i32>:18 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:19 [#uses=1]
	trunc i32 %18 to i16		; <i16>:20 [#uses=1]
	trunc i32 %16 to i16		; <i16>:21 [#uses=1]
	call i16 @llvm.atomic.cmp.swap.i16.p0i16( i16* %19, i16 %20, i16 %21 )		; <i16>:22 [#uses=1]
	store i16 %22, i16* @ss, align 2
	load i8* @sc, align 1		; <i8>:23 [#uses=1]
	sext i8 %23 to i16		; <i16>:24 [#uses=1]
	zext i16 %24 to i32		; <i32>:25 [#uses=1]
	load i8* @uc, align 1		; <i8>:26 [#uses=1]
	zext i8 %26 to i32		; <i32>:27 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:28 [#uses=1]
	trunc i32 %27 to i16		; <i16>:29 [#uses=1]
	trunc i32 %25 to i16		; <i16>:30 [#uses=1]
	call i16 @llvm.atomic.cmp.swap.i16.p0i16( i16* %28, i16 %29, i16 %30 )		; <i16>:31 [#uses=1]
	store i16 %31, i16* @us, align 2
	load i8* @sc, align 1		; <i8>:32 [#uses=1]
	sext i8 %32 to i32		; <i32>:33 [#uses=1]
	load i8* @uc, align 1		; <i8>:34 [#uses=1]
	zext i8 %34 to i32		; <i32>:35 [#uses=1]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:36 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %36, i32 %35, i32 %33 )		; <i32>:37 [#uses=1]
	store i32 %37, i32* @si, align 4
	load i8* @sc, align 1		; <i8>:38 [#uses=1]
	sext i8 %38 to i32		; <i32>:39 [#uses=1]
	load i8* @uc, align 1		; <i8>:40 [#uses=1]
	zext i8 %40 to i32		; <i32>:41 [#uses=1]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:42 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %42, i32 %41, i32 %39 )		; <i32>:43 [#uses=1]
	store i32 %43, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:44 [#uses=1]
	sext i8 %44 to i32		; <i32>:45 [#uses=1]
	load i8* @uc, align 1		; <i8>:46 [#uses=1]
	zext i8 %46 to i32		; <i32>:47 [#uses=1]
	bitcast i8* bitcast (i32* @sl to i8*) to i32*		; <i32*>:48 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %48, i32 %47, i32 %45 )		; <i32>:49 [#uses=1]
	store i32 %49, i32* @sl, align 4
	load i8* @sc, align 1		; <i8>:50 [#uses=1]
	sext i8 %50 to i32		; <i32>:51 [#uses=1]
	load i8* @uc, align 1		; <i8>:52 [#uses=1]
	zext i8 %52 to i32		; <i32>:53 [#uses=1]
	bitcast i8* bitcast (i32* @ul to i8*) to i32*		; <i32*>:54 [#uses=1]
	call i32 @llvm.atomic.cmp.swap.i32.p0i32( i32* %54, i32 %53, i32 %51 )		; <i32>:55 [#uses=1]
	store i32 %55, i32* @ul, align 4
	load i8* @sc, align 1		; <i8>:56 [#uses=1]
	zext i8 %56 to i32		; <i32>:57 [#uses=1]
	load i8* @uc, align 1		; <i8>:58 [#uses=1]
	zext i8 %58 to i32		; <i32>:59 [#uses=1]
	trunc i32 %59 to i8		; <i8>:60 [#uses=2]
	trunc i32 %57 to i8		; <i8>:61 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @sc, i8 %60, i8 %61 )		; <i8>:62 [#uses=1]
	icmp eq i8 %62, %60		; <i1>:63 [#uses=1]
	zext i1 %63 to i8		; <i8>:64 [#uses=1]
	zext i8 %64 to i32		; <i32>:65 [#uses=1]
	store i32 %65, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:66 [#uses=1]
	zext i8 %66 to i32		; <i32>:67 [#uses=1]
	load i8* @uc, align 1		; <i8>:68 [#uses=1]
	zext i8 %68 to i32		; <i32>:69 [#uses=1]
	trunc i32 %69 to i8		; <i8>:70 [#uses=2]
	trunc i32 %67 to i8		; <i8>:71 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @uc, i8 %70, i8 %71 )		; <i8>:72 [#uses=1]
	icmp eq i8 %72, %70		; <i1>:73 [#uses=1]
	zext i1 %73 to i8		; <i8>:74 [#uses=1]
	zext i8 %74 to i32		; <i32>:75 [#uses=1]
	store i32 %75, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:76 [#uses=1]
	sext i8 %76 to i16		; <i16>:77 [#uses=1]
	zext i16 %77 to i32		; <i32>:78 [#uses=1]
	load i8* @uc, align 1		; <i8>:79 [#uses=1]
	zext i8 %79 to i32		; <i32>:80 [#uses=1]
	trunc i32 %80 to i8		; <i8>:81 [#uses=2]
	trunc i32 %78 to i8		; <i8>:82 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i16* @ss to i8*), i8 %81, i8 %82 )		; <i8>:83 [#uses=1]
	icmp eq i8 %83, %81		; <i1>:84 [#uses=1]
	zext i1 %84 to i8		; <i8>:85 [#uses=1]
	zext i8 %85 to i32		; <i32>:86 [#uses=1]
	store i32 %86, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:87 [#uses=1]
	sext i8 %87 to i16		; <i16>:88 [#uses=1]
	zext i16 %88 to i32		; <i32>:89 [#uses=1]
	load i8* @uc, align 1		; <i8>:90 [#uses=1]
	zext i8 %90 to i32		; <i32>:91 [#uses=1]
	trunc i32 %91 to i8		; <i8>:92 [#uses=2]
	trunc i32 %89 to i8		; <i8>:93 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i16* @us to i8*), i8 %92, i8 %93 )		; <i8>:94 [#uses=1]
	icmp eq i8 %94, %92		; <i1>:95 [#uses=1]
	zext i1 %95 to i8		; <i8>:96 [#uses=1]
	zext i8 %96 to i32		; <i32>:97 [#uses=1]
	store i32 %97, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:98 [#uses=1]
	sext i8 %98 to i32		; <i32>:99 [#uses=1]
	load i8* @uc, align 1		; <i8>:100 [#uses=1]
	zext i8 %100 to i32		; <i32>:101 [#uses=1]
	trunc i32 %101 to i8		; <i8>:102 [#uses=2]
	trunc i32 %99 to i8		; <i8>:103 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i32* @si to i8*), i8 %102, i8 %103 )		; <i8>:104 [#uses=1]
	icmp eq i8 %104, %102		; <i1>:105 [#uses=1]
	zext i1 %105 to i8		; <i8>:106 [#uses=1]
	zext i8 %106 to i32		; <i32>:107 [#uses=1]
	store i32 %107, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:108 [#uses=1]
	sext i8 %108 to i32		; <i32>:109 [#uses=1]
	load i8* @uc, align 1		; <i8>:110 [#uses=1]
	zext i8 %110 to i32		; <i32>:111 [#uses=1]
	trunc i32 %111 to i8		; <i8>:112 [#uses=2]
	trunc i32 %109 to i8		; <i8>:113 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i32* @ui to i8*), i8 %112, i8 %113 )		; <i8>:114 [#uses=1]
	icmp eq i8 %114, %112		; <i1>:115 [#uses=1]
	zext i1 %115 to i8		; <i8>:116 [#uses=1]
	zext i8 %116 to i32		; <i32>:117 [#uses=1]
	store i32 %117, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:118 [#uses=1]
	sext i8 %118 to i32		; <i32>:119 [#uses=1]
	load i8* @uc, align 1		; <i8>:120 [#uses=1]
	zext i8 %120 to i32		; <i32>:121 [#uses=1]
	trunc i32 %121 to i8		; <i8>:122 [#uses=2]
	trunc i32 %119 to i8		; <i8>:123 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i32* @sl to i8*), i8 %122, i8 %123 )		; <i8>:124 [#uses=1]
	icmp eq i8 %124, %122		; <i1>:125 [#uses=1]
	zext i1 %125 to i8		; <i8>:126 [#uses=1]
	zext i8 %126 to i32		; <i32>:127 [#uses=1]
	store i32 %127, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:128 [#uses=1]
	sext i8 %128 to i32		; <i32>:129 [#uses=1]
	load i8* @uc, align 1		; <i8>:130 [#uses=1]
	zext i8 %130 to i32		; <i32>:131 [#uses=1]
	trunc i32 %131 to i8		; <i8>:132 [#uses=2]
	trunc i32 %129 to i8		; <i8>:133 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i32* @ul to i8*), i8 %132, i8 %133 )		; <i8>:134 [#uses=1]
	icmp eq i8 %134, %132		; <i1>:135 [#uses=1]
	zext i1 %135 to i8		; <i8>:136 [#uses=1]
	zext i8 %136 to i32		; <i32>:137 [#uses=1]
	store i32 %137, i32* @ui, align 4
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
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.swap.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.swap.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.swap.i32.p0i32(i32*, i32) nounwind

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind
