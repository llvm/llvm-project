; RUN: llvm-as < %s | opt -dse | llvm-dis

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
	%struct.cab_archive = type { i32, i16, i16, i16, i16, i8, %struct.cab_folder*, %struct.cab_file* }
	%struct.cab_file = type { i32, i16, i64, i8*, i32, i32, i32, %struct.cab_folder*, %struct.cab_file*, %struct.cab_archive*, %struct.cab_state* }
	%struct.cab_folder = type { i16, i16, %struct.cab_archive*, i64, %struct.cab_folder* }
	%struct.cab_state = type { i8*, i8*, [38912 x i8], i16, i16, i8*, i16 }
	%struct.lzx_stream = type { i32, i32, i8, i64, i64, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i8, i8, i8, i8, i32, i8*, i8*, i8*, i8*, i8*, i32, i32, i32, [84 x i8], [720 x i8], [314 x i8], [72 x i8], [104 x i16], [5408 x i16], [4596 x i16], [144 x i16], [51 x i32], [51 x i8], [32768 x i8], %struct.cab_file*, i32 (%struct.cab_file*, i8*, i32)* }

declare fastcc i32 @lzx_read_lens(%struct.lzx_stream*, i8*, i32, i32) nounwind

define i32 @lzx_decompress(%struct.lzx_stream* %lzx, i64 %out_bytes) nounwind {
bb13:		; preds = %entry
	%0 = getelementptr %struct.lzx_stream* %lzx, i32 0, i32 25		; <i8**> [#uses=2]
	%1 = getelementptr %struct.lzx_stream* %lzx, i32 0, i32 26		; <i8**> [#uses=2]
	%2 = getelementptr %struct.lzx_stream* %lzx, i32 0, i32 29		; <i32*> [#uses=0]
	br label %bb14

bb14:		; preds = %bb13
	%3 = load i8** %0, align 4		; <i8*> [#uses=1]
	%4 = load i8** %1, align 4		; <i8*> [#uses=1]
	store i8* %3, i8** %0, align 4
	store i8* %4, i8** %1, align 4
	%5 = call fastcc i32 @lzx_read_lens(%struct.lzx_stream* %lzx, i8* null, i32 256, i32 0) nounwind		; <i32> [#uses=0]
	unreachable
}
