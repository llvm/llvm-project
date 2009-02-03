; RUN: llvm-as < %s | llc -march=x86-64
; ModuleID = 'Atomics.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"
@sc = common global i8 0		; <i8*> [#uses=56]
@uc = common global i8 0		; <i8*> [#uses=116]
@ss = common global i16 0		; <i16*> [#uses=15]
@us = common global i16 0		; <i16*> [#uses=15]
@si = common global i32 0		; <i32*> [#uses=15]
@ui = common global i32 0		; <i32*> [#uses=25]
@sl = common global i64 0		; <i64*> [#uses=15]
@ul = common global i64 0		; <i64*> [#uses=15]
@sll = common global i64 0		; <i64*> [#uses=15]
@ull = common global i64 0		; <i64*> [#uses=15]

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
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:10 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %10, i64 1 )		; <i64>:11 [#uses=0]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:12 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %12, i64 1 )		; <i64>:13 [#uses=0]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:14 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %14, i64 1 )		; <i64>:15 [#uses=0]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:16 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %16, i64 1 )		; <i64>:17 [#uses=0]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @sc, i8 1 )		; <i8>:18 [#uses=0]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @uc, i8 1 )		; <i8>:19 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:20 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %20, i16 1 )		; <i16>:21 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:22 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %22, i16 1 )		; <i16>:23 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:24 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %24, i32 1 )		; <i32>:25 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:26 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %26, i32 1 )		; <i32>:27 [#uses=0]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:28 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %28, i64 1 )		; <i64>:29 [#uses=0]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:30 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %30, i64 1 )		; <i64>:31 [#uses=0]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:32 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %32, i64 1 )		; <i64>:33 [#uses=0]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:34 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %34, i64 1 )		; <i64>:35 [#uses=0]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @sc, i8 1 )		; <i8>:36 [#uses=0]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @uc, i8 1 )		; <i8>:37 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:38 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %38, i16 1 )		; <i16>:39 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:40 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %40, i16 1 )		; <i16>:41 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:42 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %42, i32 1 )		; <i32>:43 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:44 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %44, i32 1 )		; <i32>:45 [#uses=0]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:46 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %46, i64 1 )		; <i64>:47 [#uses=0]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:48 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %48, i64 1 )		; <i64>:49 [#uses=0]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:50 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %50, i64 1 )		; <i64>:51 [#uses=0]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:52 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %52, i64 1 )		; <i64>:53 [#uses=0]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @sc, i8 1 )		; <i8>:54 [#uses=0]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @uc, i8 1 )		; <i8>:55 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:56 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %56, i16 1 )		; <i16>:57 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:58 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %58, i16 1 )		; <i16>:59 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:60 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %60, i32 1 )		; <i32>:61 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:62 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %62, i32 1 )		; <i32>:63 [#uses=0]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:64 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %64, i64 1 )		; <i64>:65 [#uses=0]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:66 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %66, i64 1 )		; <i64>:67 [#uses=0]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:68 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %68, i64 1 )		; <i64>:69 [#uses=0]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:70 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %70, i64 1 )		; <i64>:71 [#uses=0]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @sc, i8 1 )		; <i8>:72 [#uses=0]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @uc, i8 1 )		; <i8>:73 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:74 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %74, i16 1 )		; <i16>:75 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:76 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %76, i16 1 )		; <i16>:77 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:78 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %78, i32 1 )		; <i32>:79 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:80 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %80, i32 1 )		; <i32>:81 [#uses=0]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:82 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %82, i64 1 )		; <i64>:83 [#uses=0]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:84 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %84, i64 1 )		; <i64>:85 [#uses=0]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:86 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %86, i64 1 )		; <i64>:87 [#uses=0]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:88 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %88, i64 1 )		; <i64>:89 [#uses=0]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @sc, i8 1 )		; <i8>:90 [#uses=0]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @uc, i8 1 )		; <i8>:91 [#uses=0]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:92 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %92, i16 1 )		; <i16>:93 [#uses=0]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:94 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %94, i16 1 )		; <i16>:95 [#uses=0]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:96 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %96, i32 1 )		; <i32>:97 [#uses=0]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:98 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %98, i32 1 )		; <i32>:99 [#uses=0]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:100 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %100, i64 1 )		; <i64>:101 [#uses=0]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:102 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %102, i64 1 )		; <i64>:103 [#uses=0]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:104 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %104, i64 1 )		; <i64>:105 [#uses=0]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:106 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %106, i64 1 )		; <i64>:107 [#uses=0]
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
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:10 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %10, i64 11 )		; <i64>:11 [#uses=1]
	store i64 %11, i64* @sl, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:12 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %12, i64 11 )		; <i64>:13 [#uses=1]
	store i64 %13, i64* @ul, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:14 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %14, i64 11 )		; <i64>:15 [#uses=1]
	store i64 %15, i64* @sll, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:16 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %16, i64 11 )		; <i64>:17 [#uses=1]
	store i64 %17, i64* @ull, align 8
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @sc, i8 11 )		; <i8>:18 [#uses=1]
	store i8 %18, i8* @sc, align 1
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @uc, i8 11 )		; <i8>:19 [#uses=1]
	store i8 %19, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:20 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %20, i16 11 )		; <i16>:21 [#uses=1]
	store i16 %21, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:22 [#uses=1]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %22, i16 11 )		; <i16>:23 [#uses=1]
	store i16 %23, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:24 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %24, i32 11 )		; <i32>:25 [#uses=1]
	store i32 %25, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:26 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %26, i32 11 )		; <i32>:27 [#uses=1]
	store i32 %27, i32* @ui, align 4
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:28 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %28, i64 11 )		; <i64>:29 [#uses=1]
	store i64 %29, i64* @sl, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:30 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %30, i64 11 )		; <i64>:31 [#uses=1]
	store i64 %31, i64* @ul, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:32 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %32, i64 11 )		; <i64>:33 [#uses=1]
	store i64 %33, i64* @sll, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:34 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %34, i64 11 )		; <i64>:35 [#uses=1]
	store i64 %35, i64* @ull, align 8
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @sc, i8 11 )		; <i8>:36 [#uses=1]
	store i8 %36, i8* @sc, align 1
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @uc, i8 11 )		; <i8>:37 [#uses=1]
	store i8 %37, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:38 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %38, i16 11 )		; <i16>:39 [#uses=1]
	store i16 %39, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:40 [#uses=1]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %40, i16 11 )		; <i16>:41 [#uses=1]
	store i16 %41, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:42 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %42, i32 11 )		; <i32>:43 [#uses=1]
	store i32 %43, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:44 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %44, i32 11 )		; <i32>:45 [#uses=1]
	store i32 %45, i32* @ui, align 4
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:46 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %46, i64 11 )		; <i64>:47 [#uses=1]
	store i64 %47, i64* @sl, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:48 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %48, i64 11 )		; <i64>:49 [#uses=1]
	store i64 %49, i64* @ul, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:50 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %50, i64 11 )		; <i64>:51 [#uses=1]
	store i64 %51, i64* @sll, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:52 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %52, i64 11 )		; <i64>:53 [#uses=1]
	store i64 %53, i64* @ull, align 8
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @sc, i8 11 )		; <i8>:54 [#uses=1]
	store i8 %54, i8* @sc, align 1
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @uc, i8 11 )		; <i8>:55 [#uses=1]
	store i8 %55, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:56 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %56, i16 11 )		; <i16>:57 [#uses=1]
	store i16 %57, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:58 [#uses=1]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %58, i16 11 )		; <i16>:59 [#uses=1]
	store i16 %59, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:60 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %60, i32 11 )		; <i32>:61 [#uses=1]
	store i32 %61, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:62 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %62, i32 11 )		; <i32>:63 [#uses=1]
	store i32 %63, i32* @ui, align 4
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:64 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %64, i64 11 )		; <i64>:65 [#uses=1]
	store i64 %65, i64* @sl, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:66 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %66, i64 11 )		; <i64>:67 [#uses=1]
	store i64 %67, i64* @ul, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:68 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %68, i64 11 )		; <i64>:69 [#uses=1]
	store i64 %69, i64* @sll, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:70 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %70, i64 11 )		; <i64>:71 [#uses=1]
	store i64 %71, i64* @ull, align 8
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @sc, i8 11 )		; <i8>:72 [#uses=1]
	store i8 %72, i8* @sc, align 1
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @uc, i8 11 )		; <i8>:73 [#uses=1]
	store i8 %73, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:74 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %74, i16 11 )		; <i16>:75 [#uses=1]
	store i16 %75, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:76 [#uses=1]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %76, i16 11 )		; <i16>:77 [#uses=1]
	store i16 %77, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:78 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %78, i32 11 )		; <i32>:79 [#uses=1]
	store i32 %79, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:80 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %80, i32 11 )		; <i32>:81 [#uses=1]
	store i32 %81, i32* @ui, align 4
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:82 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %82, i64 11 )		; <i64>:83 [#uses=1]
	store i64 %83, i64* @sl, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:84 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %84, i64 11 )		; <i64>:85 [#uses=1]
	store i64 %85, i64* @ul, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:86 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %86, i64 11 )		; <i64>:87 [#uses=1]
	store i64 %87, i64* @sll, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:88 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %88, i64 11 )		; <i64>:89 [#uses=1]
	store i64 %89, i64* @ull, align 8
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @sc, i8 11 )		; <i8>:90 [#uses=1]
	store i8 %90, i8* @sc, align 1
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @uc, i8 11 )		; <i8>:91 [#uses=1]
	store i8 %91, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:92 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %92, i16 11 )		; <i16>:93 [#uses=1]
	store i16 %93, i16* @ss, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:94 [#uses=1]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %94, i16 11 )		; <i16>:95 [#uses=1]
	store i16 %95, i16* @us, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:96 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %96, i32 11 )		; <i32>:97 [#uses=1]
	store i32 %97, i32* @si, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:98 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %98, i32 11 )		; <i32>:99 [#uses=1]
	store i32 %99, i32* @ui, align 4
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:100 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %100, i64 11 )		; <i64>:101 [#uses=1]
	store i64 %101, i64* @sl, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:102 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %102, i64 11 )		; <i64>:103 [#uses=1]
	store i64 %103, i64* @ul, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:104 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %104, i64 11 )		; <i64>:105 [#uses=1]
	store i64 %105, i64* @sll, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:106 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %106, i64 11 )		; <i64>:107 [#uses=1]
	store i64 %107, i64* @ull, align 8
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
	zext i8 %32 to i64		; <i64>:33 [#uses=2]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:34 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %34, i64 %33 )		; <i64>:35 [#uses=1]
	add i64 %35, %33		; <i64>:36 [#uses=1]
	store i64 %36, i64* @sl, align 8
	load i8* @uc, align 1		; <i8>:37 [#uses=1]
	zext i8 %37 to i64		; <i64>:38 [#uses=2]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:39 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %39, i64 %38 )		; <i64>:40 [#uses=1]
	add i64 %40, %38		; <i64>:41 [#uses=1]
	store i64 %41, i64* @ul, align 8
	load i8* @uc, align 1		; <i8>:42 [#uses=1]
	zext i8 %42 to i64		; <i64>:43 [#uses=2]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:44 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %44, i64 %43 )		; <i64>:45 [#uses=1]
	add i64 %45, %43		; <i64>:46 [#uses=1]
	store i64 %46, i64* @sll, align 8
	load i8* @uc, align 1		; <i8>:47 [#uses=1]
	zext i8 %47 to i64		; <i64>:48 [#uses=2]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:49 [#uses=1]
	call i64 @llvm.atomic.load.add.i64.p0i64( i64* %49, i64 %48 )		; <i64>:50 [#uses=1]
	add i64 %50, %48		; <i64>:51 [#uses=1]
	store i64 %51, i64* @ull, align 8
	load i8* @uc, align 1		; <i8>:52 [#uses=1]
	zext i8 %52 to i32		; <i32>:53 [#uses=1]
	trunc i32 %53 to i8		; <i8>:54 [#uses=2]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @sc, i8 %54 )		; <i8>:55 [#uses=1]
	sub i8 %55, %54		; <i8>:56 [#uses=1]
	store i8 %56, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:57 [#uses=1]
	zext i8 %57 to i32		; <i32>:58 [#uses=1]
	trunc i32 %58 to i8		; <i8>:59 [#uses=2]
	call i8 @llvm.atomic.load.sub.i8.p0i8( i8* @uc, i8 %59 )		; <i8>:60 [#uses=1]
	sub i8 %60, %59		; <i8>:61 [#uses=1]
	store i8 %61, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:62 [#uses=1]
	zext i8 %62 to i32		; <i32>:63 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:64 [#uses=1]
	trunc i32 %63 to i16		; <i16>:65 [#uses=2]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %64, i16 %65 )		; <i16>:66 [#uses=1]
	sub i16 %66, %65		; <i16>:67 [#uses=1]
	store i16 %67, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:68 [#uses=1]
	zext i8 %68 to i32		; <i32>:69 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:70 [#uses=1]
	trunc i32 %69 to i16		; <i16>:71 [#uses=2]
	call i16 @llvm.atomic.load.sub.i16.p0i16( i16* %70, i16 %71 )		; <i16>:72 [#uses=1]
	sub i16 %72, %71		; <i16>:73 [#uses=1]
	store i16 %73, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:74 [#uses=1]
	zext i8 %74 to i32		; <i32>:75 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:76 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %76, i32 %75 )		; <i32>:77 [#uses=1]
	sub i32 %77, %75		; <i32>:78 [#uses=1]
	store i32 %78, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:79 [#uses=1]
	zext i8 %79 to i32		; <i32>:80 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:81 [#uses=1]
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %81, i32 %80 )		; <i32>:82 [#uses=1]
	sub i32 %82, %80		; <i32>:83 [#uses=1]
	store i32 %83, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:84 [#uses=1]
	zext i8 %84 to i64		; <i64>:85 [#uses=2]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:86 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %86, i64 %85 )		; <i64>:87 [#uses=1]
	sub i64 %87, %85		; <i64>:88 [#uses=1]
	store i64 %88, i64* @sl, align 8
	load i8* @uc, align 1		; <i8>:89 [#uses=1]
	zext i8 %89 to i64		; <i64>:90 [#uses=2]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:91 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %91, i64 %90 )		; <i64>:92 [#uses=1]
	sub i64 %92, %90		; <i64>:93 [#uses=1]
	store i64 %93, i64* @ul, align 8
	load i8* @uc, align 1		; <i8>:94 [#uses=1]
	zext i8 %94 to i64		; <i64>:95 [#uses=2]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:96 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %96, i64 %95 )		; <i64>:97 [#uses=1]
	sub i64 %97, %95		; <i64>:98 [#uses=1]
	store i64 %98, i64* @sll, align 8
	load i8* @uc, align 1		; <i8>:99 [#uses=1]
	zext i8 %99 to i64		; <i64>:100 [#uses=2]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:101 [#uses=1]
	call i64 @llvm.atomic.load.sub.i64.p0i64( i64* %101, i64 %100 )		; <i64>:102 [#uses=1]
	sub i64 %102, %100		; <i64>:103 [#uses=1]
	store i64 %103, i64* @ull, align 8
	load i8* @uc, align 1		; <i8>:104 [#uses=1]
	zext i8 %104 to i32		; <i32>:105 [#uses=1]
	trunc i32 %105 to i8		; <i8>:106 [#uses=2]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @sc, i8 %106 )		; <i8>:107 [#uses=1]
	or i8 %107, %106		; <i8>:108 [#uses=1]
	store i8 %108, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:109 [#uses=1]
	zext i8 %109 to i32		; <i32>:110 [#uses=1]
	trunc i32 %110 to i8		; <i8>:111 [#uses=2]
	call i8 @llvm.atomic.load.or.i8.p0i8( i8* @uc, i8 %111 )		; <i8>:112 [#uses=1]
	or i8 %112, %111		; <i8>:113 [#uses=1]
	store i8 %113, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:114 [#uses=1]
	zext i8 %114 to i32		; <i32>:115 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:116 [#uses=1]
	trunc i32 %115 to i16		; <i16>:117 [#uses=2]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %116, i16 %117 )		; <i16>:118 [#uses=1]
	or i16 %118, %117		; <i16>:119 [#uses=1]
	store i16 %119, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:120 [#uses=1]
	zext i8 %120 to i32		; <i32>:121 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:122 [#uses=1]
	trunc i32 %121 to i16		; <i16>:123 [#uses=2]
	call i16 @llvm.atomic.load.or.i16.p0i16( i16* %122, i16 %123 )		; <i16>:124 [#uses=1]
	or i16 %124, %123		; <i16>:125 [#uses=1]
	store i16 %125, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:126 [#uses=1]
	zext i8 %126 to i32		; <i32>:127 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:128 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %128, i32 %127 )		; <i32>:129 [#uses=1]
	or i32 %129, %127		; <i32>:130 [#uses=1]
	store i32 %130, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:131 [#uses=1]
	zext i8 %131 to i32		; <i32>:132 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:133 [#uses=1]
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %133, i32 %132 )		; <i32>:134 [#uses=1]
	or i32 %134, %132		; <i32>:135 [#uses=1]
	store i32 %135, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:136 [#uses=1]
	zext i8 %136 to i64		; <i64>:137 [#uses=2]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:138 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %138, i64 %137 )		; <i64>:139 [#uses=1]
	or i64 %139, %137		; <i64>:140 [#uses=1]
	store i64 %140, i64* @sl, align 8
	load i8* @uc, align 1		; <i8>:141 [#uses=1]
	zext i8 %141 to i64		; <i64>:142 [#uses=2]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:143 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %143, i64 %142 )		; <i64>:144 [#uses=1]
	or i64 %144, %142		; <i64>:145 [#uses=1]
	store i64 %145, i64* @ul, align 8
	load i8* @uc, align 1		; <i8>:146 [#uses=1]
	zext i8 %146 to i64		; <i64>:147 [#uses=2]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:148 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %148, i64 %147 )		; <i64>:149 [#uses=1]
	or i64 %149, %147		; <i64>:150 [#uses=1]
	store i64 %150, i64* @sll, align 8
	load i8* @uc, align 1		; <i8>:151 [#uses=1]
	zext i8 %151 to i64		; <i64>:152 [#uses=2]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:153 [#uses=1]
	call i64 @llvm.atomic.load.or.i64.p0i64( i64* %153, i64 %152 )		; <i64>:154 [#uses=1]
	or i64 %154, %152		; <i64>:155 [#uses=1]
	store i64 %155, i64* @ull, align 8
	load i8* @uc, align 1		; <i8>:156 [#uses=1]
	zext i8 %156 to i32		; <i32>:157 [#uses=1]
	trunc i32 %157 to i8		; <i8>:158 [#uses=2]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @sc, i8 %158 )		; <i8>:159 [#uses=1]
	xor i8 %159, %158		; <i8>:160 [#uses=1]
	store i8 %160, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:161 [#uses=1]
	zext i8 %161 to i32		; <i32>:162 [#uses=1]
	trunc i32 %162 to i8		; <i8>:163 [#uses=2]
	call i8 @llvm.atomic.load.xor.i8.p0i8( i8* @uc, i8 %163 )		; <i8>:164 [#uses=1]
	xor i8 %164, %163		; <i8>:165 [#uses=1]
	store i8 %165, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:166 [#uses=1]
	zext i8 %166 to i32		; <i32>:167 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:168 [#uses=1]
	trunc i32 %167 to i16		; <i16>:169 [#uses=2]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %168, i16 %169 )		; <i16>:170 [#uses=1]
	xor i16 %170, %169		; <i16>:171 [#uses=1]
	store i16 %171, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:172 [#uses=1]
	zext i8 %172 to i32		; <i32>:173 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:174 [#uses=1]
	trunc i32 %173 to i16		; <i16>:175 [#uses=2]
	call i16 @llvm.atomic.load.xor.i16.p0i16( i16* %174, i16 %175 )		; <i16>:176 [#uses=1]
	xor i16 %176, %175		; <i16>:177 [#uses=1]
	store i16 %177, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:178 [#uses=1]
	zext i8 %178 to i32		; <i32>:179 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:180 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %180, i32 %179 )		; <i32>:181 [#uses=1]
	xor i32 %181, %179		; <i32>:182 [#uses=1]
	store i32 %182, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:183 [#uses=1]
	zext i8 %183 to i32		; <i32>:184 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:185 [#uses=1]
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %185, i32 %184 )		; <i32>:186 [#uses=1]
	xor i32 %186, %184		; <i32>:187 [#uses=1]
	store i32 %187, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:188 [#uses=1]
	zext i8 %188 to i64		; <i64>:189 [#uses=2]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:190 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %190, i64 %189 )		; <i64>:191 [#uses=1]
	xor i64 %191, %189		; <i64>:192 [#uses=1]
	store i64 %192, i64* @sl, align 8
	load i8* @uc, align 1		; <i8>:193 [#uses=1]
	zext i8 %193 to i64		; <i64>:194 [#uses=2]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:195 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %195, i64 %194 )		; <i64>:196 [#uses=1]
	xor i64 %196, %194		; <i64>:197 [#uses=1]
	store i64 %197, i64* @ul, align 8
	load i8* @uc, align 1		; <i8>:198 [#uses=1]
	zext i8 %198 to i64		; <i64>:199 [#uses=2]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:200 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %200, i64 %199 )		; <i64>:201 [#uses=1]
	xor i64 %201, %199		; <i64>:202 [#uses=1]
	store i64 %202, i64* @sll, align 8
	load i8* @uc, align 1		; <i8>:203 [#uses=1]
	zext i8 %203 to i64		; <i64>:204 [#uses=2]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:205 [#uses=1]
	call i64 @llvm.atomic.load.xor.i64.p0i64( i64* %205, i64 %204 )		; <i64>:206 [#uses=1]
	xor i64 %206, %204		; <i64>:207 [#uses=1]
	store i64 %207, i64* @ull, align 8
	load i8* @uc, align 1		; <i8>:208 [#uses=1]
	zext i8 %208 to i32		; <i32>:209 [#uses=1]
	trunc i32 %209 to i8		; <i8>:210 [#uses=2]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @sc, i8 %210 )		; <i8>:211 [#uses=1]
	and i8 %211, %210		; <i8>:212 [#uses=1]
	store i8 %212, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:213 [#uses=1]
	zext i8 %213 to i32		; <i32>:214 [#uses=1]
	trunc i32 %214 to i8		; <i8>:215 [#uses=2]
	call i8 @llvm.atomic.load.and.i8.p0i8( i8* @uc, i8 %215 )		; <i8>:216 [#uses=1]
	and i8 %216, %215		; <i8>:217 [#uses=1]
	store i8 %217, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:218 [#uses=1]
	zext i8 %218 to i32		; <i32>:219 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:220 [#uses=1]
	trunc i32 %219 to i16		; <i16>:221 [#uses=2]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %220, i16 %221 )		; <i16>:222 [#uses=1]
	and i16 %222, %221		; <i16>:223 [#uses=1]
	store i16 %223, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:224 [#uses=1]
	zext i8 %224 to i32		; <i32>:225 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:226 [#uses=1]
	trunc i32 %225 to i16		; <i16>:227 [#uses=2]
	call i16 @llvm.atomic.load.and.i16.p0i16( i16* %226, i16 %227 )		; <i16>:228 [#uses=1]
	and i16 %228, %227		; <i16>:229 [#uses=1]
	store i16 %229, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:230 [#uses=1]
	zext i8 %230 to i32		; <i32>:231 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:232 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %232, i32 %231 )		; <i32>:233 [#uses=1]
	and i32 %233, %231		; <i32>:234 [#uses=1]
	store i32 %234, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:235 [#uses=1]
	zext i8 %235 to i32		; <i32>:236 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:237 [#uses=1]
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %237, i32 %236 )		; <i32>:238 [#uses=1]
	and i32 %238, %236		; <i32>:239 [#uses=1]
	store i32 %239, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:240 [#uses=1]
	zext i8 %240 to i64		; <i64>:241 [#uses=2]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:242 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %242, i64 %241 )		; <i64>:243 [#uses=1]
	and i64 %243, %241		; <i64>:244 [#uses=1]
	store i64 %244, i64* @sl, align 8
	load i8* @uc, align 1		; <i8>:245 [#uses=1]
	zext i8 %245 to i64		; <i64>:246 [#uses=2]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:247 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %247, i64 %246 )		; <i64>:248 [#uses=1]
	and i64 %248, %246		; <i64>:249 [#uses=1]
	store i64 %249, i64* @ul, align 8
	load i8* @uc, align 1		; <i8>:250 [#uses=1]
	zext i8 %250 to i64		; <i64>:251 [#uses=2]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:252 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %252, i64 %251 )		; <i64>:253 [#uses=1]
	and i64 %253, %251		; <i64>:254 [#uses=1]
	store i64 %254, i64* @sll, align 8
	load i8* @uc, align 1		; <i8>:255 [#uses=1]
	zext i8 %255 to i64		; <i64>:256 [#uses=2]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:257 [#uses=1]
	call i64 @llvm.atomic.load.and.i64.p0i64( i64* %257, i64 %256 )		; <i64>:258 [#uses=1]
	and i64 %258, %256		; <i64>:259 [#uses=1]
	store i64 %259, i64* @ull, align 8
	load i8* @uc, align 1		; <i8>:260 [#uses=1]
	zext i8 %260 to i32		; <i32>:261 [#uses=1]
	trunc i32 %261 to i8		; <i8>:262 [#uses=2]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @sc, i8 %262 )		; <i8>:263 [#uses=1]
	xor i8 %263, -1		; <i8>:264 [#uses=1]
	and i8 %264, %262		; <i8>:265 [#uses=1]
	store i8 %265, i8* @sc, align 1
	load i8* @uc, align 1		; <i8>:266 [#uses=1]
	zext i8 %266 to i32		; <i32>:267 [#uses=1]
	trunc i32 %267 to i8		; <i8>:268 [#uses=2]
	call i8 @llvm.atomic.load.nand.i8.p0i8( i8* @uc, i8 %268 )		; <i8>:269 [#uses=1]
	xor i8 %269, -1		; <i8>:270 [#uses=1]
	and i8 %270, %268		; <i8>:271 [#uses=1]
	store i8 %271, i8* @uc, align 1
	load i8* @uc, align 1		; <i8>:272 [#uses=1]
	zext i8 %272 to i32		; <i32>:273 [#uses=1]
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:274 [#uses=1]
	trunc i32 %273 to i16		; <i16>:275 [#uses=2]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %274, i16 %275 )		; <i16>:276 [#uses=1]
	xor i16 %276, -1		; <i16>:277 [#uses=1]
	and i16 %277, %275		; <i16>:278 [#uses=1]
	store i16 %278, i16* @ss, align 2
	load i8* @uc, align 1		; <i8>:279 [#uses=1]
	zext i8 %279 to i32		; <i32>:280 [#uses=1]
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:281 [#uses=1]
	trunc i32 %280 to i16		; <i16>:282 [#uses=2]
	call i16 @llvm.atomic.load.nand.i16.p0i16( i16* %281, i16 %282 )		; <i16>:283 [#uses=1]
	xor i16 %283, -1		; <i16>:284 [#uses=1]
	and i16 %284, %282		; <i16>:285 [#uses=1]
	store i16 %285, i16* @us, align 2
	load i8* @uc, align 1		; <i8>:286 [#uses=1]
	zext i8 %286 to i32		; <i32>:287 [#uses=2]
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:288 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %288, i32 %287 )		; <i32>:289 [#uses=1]
	xor i32 %289, -1		; <i32>:290 [#uses=1]
	and i32 %290, %287		; <i32>:291 [#uses=1]
	store i32 %291, i32* @si, align 4
	load i8* @uc, align 1		; <i8>:292 [#uses=1]
	zext i8 %292 to i32		; <i32>:293 [#uses=2]
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:294 [#uses=1]
	call i32 @llvm.atomic.load.nand.i32.p0i32( i32* %294, i32 %293 )		; <i32>:295 [#uses=1]
	xor i32 %295, -1		; <i32>:296 [#uses=1]
	and i32 %296, %293		; <i32>:297 [#uses=1]
	store i32 %297, i32* @ui, align 4
	load i8* @uc, align 1		; <i8>:298 [#uses=1]
	zext i8 %298 to i64		; <i64>:299 [#uses=2]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:300 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %300, i64 %299 )		; <i64>:301 [#uses=1]
	xor i64 %301, -1		; <i64>:302 [#uses=1]
	and i64 %302, %299		; <i64>:303 [#uses=1]
	store i64 %303, i64* @sl, align 8
	load i8* @uc, align 1		; <i8>:304 [#uses=1]
	zext i8 %304 to i64		; <i64>:305 [#uses=2]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:306 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %306, i64 %305 )		; <i64>:307 [#uses=1]
	xor i64 %307, -1		; <i64>:308 [#uses=1]
	and i64 %308, %305		; <i64>:309 [#uses=1]
	store i64 %309, i64* @ul, align 8
	load i8* @uc, align 1		; <i8>:310 [#uses=1]
	zext i8 %310 to i64		; <i64>:311 [#uses=2]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:312 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %312, i64 %311 )		; <i64>:313 [#uses=1]
	xor i64 %313, -1		; <i64>:314 [#uses=1]
	and i64 %314, %311		; <i64>:315 [#uses=1]
	store i64 %315, i64* @sll, align 8
	load i8* @uc, align 1		; <i8>:316 [#uses=1]
	zext i8 %316 to i64		; <i64>:317 [#uses=2]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:318 [#uses=1]
	call i64 @llvm.atomic.load.nand.i64.p0i64( i64* %318, i64 %317 )		; <i64>:319 [#uses=1]
	xor i64 %319, -1		; <i64>:320 [#uses=1]
	and i64 %320, %317		; <i64>:321 [#uses=1]
	store i64 %321, i64* @ull, align 8
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
	sext i8 %44 to i64		; <i64>:45 [#uses=1]
	load i8* @uc, align 1		; <i8>:46 [#uses=1]
	zext i8 %46 to i64		; <i64>:47 [#uses=1]
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:48 [#uses=1]
	call i64 @llvm.atomic.cmp.swap.i64.p0i64( i64* %48, i64 %47, i64 %45 )		; <i64>:49 [#uses=1]
	store i64 %49, i64* @sl, align 8
	load i8* @sc, align 1		; <i8>:50 [#uses=1]
	sext i8 %50 to i64		; <i64>:51 [#uses=1]
	load i8* @uc, align 1		; <i8>:52 [#uses=1]
	zext i8 %52 to i64		; <i64>:53 [#uses=1]
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:54 [#uses=1]
	call i64 @llvm.atomic.cmp.swap.i64.p0i64( i64* %54, i64 %53, i64 %51 )		; <i64>:55 [#uses=1]
	store i64 %55, i64* @ul, align 8
	load i8* @sc, align 1		; <i8>:56 [#uses=1]
	sext i8 %56 to i64		; <i64>:57 [#uses=1]
	load i8* @uc, align 1		; <i8>:58 [#uses=1]
	zext i8 %58 to i64		; <i64>:59 [#uses=1]
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:60 [#uses=1]
	call i64 @llvm.atomic.cmp.swap.i64.p0i64( i64* %60, i64 %59, i64 %57 )		; <i64>:61 [#uses=1]
	store i64 %61, i64* @sll, align 8
	load i8* @sc, align 1		; <i8>:62 [#uses=1]
	sext i8 %62 to i64		; <i64>:63 [#uses=1]
	load i8* @uc, align 1		; <i8>:64 [#uses=1]
	zext i8 %64 to i64		; <i64>:65 [#uses=1]
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:66 [#uses=1]
	call i64 @llvm.atomic.cmp.swap.i64.p0i64( i64* %66, i64 %65, i64 %63 )		; <i64>:67 [#uses=1]
	store i64 %67, i64* @ull, align 8
	load i8* @sc, align 1		; <i8>:68 [#uses=1]
	zext i8 %68 to i32		; <i32>:69 [#uses=1]
	load i8* @uc, align 1		; <i8>:70 [#uses=1]
	zext i8 %70 to i32		; <i32>:71 [#uses=1]
	trunc i32 %71 to i8		; <i8>:72 [#uses=2]
	trunc i32 %69 to i8		; <i8>:73 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @sc, i8 %72, i8 %73 )		; <i8>:74 [#uses=1]
	icmp eq i8 %74, %72		; <i1>:75 [#uses=1]
	zext i1 %75 to i8		; <i8>:76 [#uses=1]
	zext i8 %76 to i32		; <i32>:77 [#uses=1]
	store i32 %77, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:78 [#uses=1]
	zext i8 %78 to i32		; <i32>:79 [#uses=1]
	load i8* @uc, align 1		; <i8>:80 [#uses=1]
	zext i8 %80 to i32		; <i32>:81 [#uses=1]
	trunc i32 %81 to i8		; <i8>:82 [#uses=2]
	trunc i32 %79 to i8		; <i8>:83 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* @uc, i8 %82, i8 %83 )		; <i8>:84 [#uses=1]
	icmp eq i8 %84, %82		; <i1>:85 [#uses=1]
	zext i1 %85 to i8		; <i8>:86 [#uses=1]
	zext i8 %86 to i32		; <i32>:87 [#uses=1]
	store i32 %87, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:88 [#uses=1]
	sext i8 %88 to i16		; <i16>:89 [#uses=1]
	zext i16 %89 to i32		; <i32>:90 [#uses=1]
	load i8* @uc, align 1		; <i8>:91 [#uses=1]
	zext i8 %91 to i32		; <i32>:92 [#uses=1]
	trunc i32 %92 to i8		; <i8>:93 [#uses=2]
	trunc i32 %90 to i8		; <i8>:94 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i16* @ss to i8*), i8 %93, i8 %94 )		; <i8>:95 [#uses=1]
	icmp eq i8 %95, %93		; <i1>:96 [#uses=1]
	zext i1 %96 to i8		; <i8>:97 [#uses=1]
	zext i8 %97 to i32		; <i32>:98 [#uses=1]
	store i32 %98, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:99 [#uses=1]
	sext i8 %99 to i16		; <i16>:100 [#uses=1]
	zext i16 %100 to i32		; <i32>:101 [#uses=1]
	load i8* @uc, align 1		; <i8>:102 [#uses=1]
	zext i8 %102 to i32		; <i32>:103 [#uses=1]
	trunc i32 %103 to i8		; <i8>:104 [#uses=2]
	trunc i32 %101 to i8		; <i8>:105 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i16* @us to i8*), i8 %104, i8 %105 )		; <i8>:106 [#uses=1]
	icmp eq i8 %106, %104		; <i1>:107 [#uses=1]
	zext i1 %107 to i8		; <i8>:108 [#uses=1]
	zext i8 %108 to i32		; <i32>:109 [#uses=1]
	store i32 %109, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:110 [#uses=1]
	sext i8 %110 to i32		; <i32>:111 [#uses=1]
	load i8* @uc, align 1		; <i8>:112 [#uses=1]
	zext i8 %112 to i32		; <i32>:113 [#uses=1]
	trunc i32 %113 to i8		; <i8>:114 [#uses=2]
	trunc i32 %111 to i8		; <i8>:115 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i32* @si to i8*), i8 %114, i8 %115 )		; <i8>:116 [#uses=1]
	icmp eq i8 %116, %114		; <i1>:117 [#uses=1]
	zext i1 %117 to i8		; <i8>:118 [#uses=1]
	zext i8 %118 to i32		; <i32>:119 [#uses=1]
	store i32 %119, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:120 [#uses=1]
	sext i8 %120 to i32		; <i32>:121 [#uses=1]
	load i8* @uc, align 1		; <i8>:122 [#uses=1]
	zext i8 %122 to i32		; <i32>:123 [#uses=1]
	trunc i32 %123 to i8		; <i8>:124 [#uses=2]
	trunc i32 %121 to i8		; <i8>:125 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i32* @ui to i8*), i8 %124, i8 %125 )		; <i8>:126 [#uses=1]
	icmp eq i8 %126, %124		; <i1>:127 [#uses=1]
	zext i1 %127 to i8		; <i8>:128 [#uses=1]
	zext i8 %128 to i32		; <i32>:129 [#uses=1]
	store i32 %129, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:130 [#uses=1]
	sext i8 %130 to i64		; <i64>:131 [#uses=1]
	load i8* @uc, align 1		; <i8>:132 [#uses=1]
	zext i8 %132 to i64		; <i64>:133 [#uses=1]
	trunc i64 %133 to i8		; <i8>:134 [#uses=2]
	trunc i64 %131 to i8		; <i8>:135 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i64* @sl to i8*), i8 %134, i8 %135 )		; <i8>:136 [#uses=1]
	icmp eq i8 %136, %134		; <i1>:137 [#uses=1]
	zext i1 %137 to i8		; <i8>:138 [#uses=1]
	zext i8 %138 to i32		; <i32>:139 [#uses=1]
	store i32 %139, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:140 [#uses=1]
	sext i8 %140 to i64		; <i64>:141 [#uses=1]
	load i8* @uc, align 1		; <i8>:142 [#uses=1]
	zext i8 %142 to i64		; <i64>:143 [#uses=1]
	trunc i64 %143 to i8		; <i8>:144 [#uses=2]
	trunc i64 %141 to i8		; <i8>:145 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i64* @ul to i8*), i8 %144, i8 %145 )		; <i8>:146 [#uses=1]
	icmp eq i8 %146, %144		; <i1>:147 [#uses=1]
	zext i1 %147 to i8		; <i8>:148 [#uses=1]
	zext i8 %148 to i32		; <i32>:149 [#uses=1]
	store i32 %149, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:150 [#uses=1]
	sext i8 %150 to i64		; <i64>:151 [#uses=1]
	load i8* @uc, align 1		; <i8>:152 [#uses=1]
	zext i8 %152 to i64		; <i64>:153 [#uses=1]
	trunc i64 %153 to i8		; <i8>:154 [#uses=2]
	trunc i64 %151 to i8		; <i8>:155 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i64* @sll to i8*), i8 %154, i8 %155 )		; <i8>:156 [#uses=1]
	icmp eq i8 %156, %154		; <i1>:157 [#uses=1]
	zext i1 %157 to i8		; <i8>:158 [#uses=1]
	zext i8 %158 to i32		; <i32>:159 [#uses=1]
	store i32 %159, i32* @ui, align 4
	load i8* @sc, align 1		; <i8>:160 [#uses=1]
	sext i8 %160 to i64		; <i64>:161 [#uses=1]
	load i8* @uc, align 1		; <i8>:162 [#uses=1]
	zext i8 %162 to i64		; <i64>:163 [#uses=1]
	trunc i64 %163 to i8		; <i8>:164 [#uses=2]
	trunc i64 %161 to i8		; <i8>:165 [#uses=1]
	call i8 @llvm.atomic.cmp.swap.i8.p0i8( i8* bitcast (i64* @ull to i8*), i8 %164, i8 %165 )		; <i8>:166 [#uses=1]
	icmp eq i8 %166, %164		; <i1>:167 [#uses=1]
	zext i1 %167 to i8		; <i8>:168 [#uses=1]
	zext i8 %168 to i32		; <i32>:169 [#uses=1]
	store i32 %169, i32* @ui, align 4
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.cmp.swap.i8.p0i8(i8*, i8, i8) nounwind

declare i16 @llvm.atomic.cmp.swap.i16.p0i16(i16*, i16, i16) nounwind

declare i32 @llvm.atomic.cmp.swap.i32.p0i32(i32*, i32, i32) nounwind

declare i64 @llvm.atomic.cmp.swap.i64.p0i64(i64*, i64, i64) nounwind

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
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:10 [#uses=1]
	call i64 @llvm.atomic.swap.i64.p0i64( i64* %10, i64 1 )		; <i64>:11 [#uses=1]
	store i64 %11, i64* @sl, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:12 [#uses=1]
	call i64 @llvm.atomic.swap.i64.p0i64( i64* %12, i64 1 )		; <i64>:13 [#uses=1]
	store i64 %13, i64* @ul, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:14 [#uses=1]
	call i64 @llvm.atomic.swap.i64.p0i64( i64* %14, i64 1 )		; <i64>:15 [#uses=1]
	store i64 %15, i64* @sll, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:16 [#uses=1]
	call i64 @llvm.atomic.swap.i64.p0i64( i64* %16, i64 1 )		; <i64>:17 [#uses=1]
	store i64 %17, i64* @ull, align 8
	call void @llvm.memory.barrier( i1 true, i1 true, i1 true, i1 true, i1 false )
	volatile store i8 0, i8* @sc, align 1
	volatile store i8 0, i8* @uc, align 1
	bitcast i8* bitcast (i16* @ss to i8*) to i16*		; <i16*>:18 [#uses=1]
	volatile store i16 0, i16* %18, align 2
	bitcast i8* bitcast (i16* @us to i8*) to i16*		; <i16*>:19 [#uses=1]
	volatile store i16 0, i16* %19, align 2
	bitcast i8* bitcast (i32* @si to i8*) to i32*		; <i32*>:20 [#uses=1]
	volatile store i32 0, i32* %20, align 4
	bitcast i8* bitcast (i32* @ui to i8*) to i32*		; <i32*>:21 [#uses=1]
	volatile store i32 0, i32* %21, align 4
	bitcast i8* bitcast (i64* @sl to i8*) to i64*		; <i64*>:22 [#uses=1]
	volatile store i64 0, i64* %22, align 8
	bitcast i8* bitcast (i64* @ul to i8*) to i64*		; <i64*>:23 [#uses=1]
	volatile store i64 0, i64* %23, align 8
	bitcast i8* bitcast (i64* @sll to i8*) to i64*		; <i64*>:24 [#uses=1]
	volatile store i64 0, i64* %24, align 8
	bitcast i8* bitcast (i64* @ull to i8*) to i64*		; <i64*>:25 [#uses=1]
	volatile store i64 0, i64* %25, align 8
	br label %return

return:		; preds = %entry
	ret void
}

declare i8 @llvm.atomic.swap.i8.p0i8(i8*, i8) nounwind

declare i16 @llvm.atomic.swap.i16.p0i16(i16*, i16) nounwind

declare i32 @llvm.atomic.swap.i32.p0i32(i32*, i32) nounwind

declare i64 @llvm.atomic.swap.i64.p0i64(i64*, i64) nounwind

declare void @llvm.memory.barrier(i1, i1, i1, i1, i1) nounwind
