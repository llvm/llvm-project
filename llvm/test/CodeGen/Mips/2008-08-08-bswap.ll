; RUN: llvm-as < %s | llc -march=mips | grep wsbw | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "psp"

define i32 @__bswapsi2(i32 %u) nounwind {
entry:
	tail call i32 @llvm.bswap.i32( i32 %u )		; <i32>:0 [#uses=1]
	ret i32 %0
}

declare i32 @llvm.bswap.i32(i32) nounwind readnone
