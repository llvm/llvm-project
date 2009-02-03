; RUN: llvm-as < %s | llc -march=mips | grep {lw.*(\$4)} | count 2

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"
	%struct.byval0 = type { i32, i32 }

define i64 @test0(%struct.byval0* byval  %b, i64 %sum) nounwind  {
entry:
	getelementptr %struct.byval0* %b, i32 0, i32 0		; <i32*>:0 [#uses=1]
	load i32* %0, align 4		; <i32>:1 [#uses=1]
	getelementptr %struct.byval0* %b, i32 0, i32 1		; <i32*>:2 [#uses=1]
	load i32* %2, align 4		; <i32>:3 [#uses=1]
	add i32 %3, %1		; <i32>:4 [#uses=1]
	sext i32 %4 to i64		; <i64>:5 [#uses=1]
	add i64 %5, %sum		; <i64>:6 [#uses=1]
	ret i64 %6
}

