; RUN: llc < %s -mtriple=i386-unknown-linux-gnu
; PR3995

        %struct.vtable = type { ptr }
	%struct.array = type { %struct.impl, [256 x %struct.pair], [256 x %struct.pair], [256 x %struct.pair], [256 x %struct.pair], [256 x %struct.pair], [256 x %struct.pair] }
	%struct.impl = type { %struct.vtable, i8, ptr, i32, i32, i64, i64 }
	%struct.pair = type { i64, i64 }

define void @test() {
entry:
	%0 = load i32, ptr null, align 4		; <i32> [#uses=1]
	%1 = lshr i32 %0, 8		; <i32> [#uses=1]
	%2 = and i32 %1, 255		; <i32> [#uses=1]
	%3 = getelementptr %struct.array, ptr null, i32 0, i32 3		; <ptr> [#uses=1]
	%4 = getelementptr [256 x %struct.pair], ptr %3, i32 0, i32 %2		; <ptr> [#uses=1]
	%5 = getelementptr %struct.pair, ptr %4, i32 0, i32 1		; <ptr> [#uses=1]
	%6 = load i64, ptr %5, align 4		; <i64> [#uses=1]
	%7 = xor i64 0, %6		; <i64> [#uses=1]
	%8 = xor i64 %7, 0		; <i64> [#uses=1]
	%9 = xor i64 %8, 0		; <i64> [#uses=1]
	store i64 %9, ptr null, align 8
	unreachable
}
