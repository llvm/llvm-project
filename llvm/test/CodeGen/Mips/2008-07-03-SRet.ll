; RUN: llvm-as < %s | llc -march=mips | grep {sw.*(\$4)} | count 3

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "mipsallegrexel-psp-elf"
	%struct.sret0 = type { i32, i32, i32 }

define void @test0(%struct.sret0* noalias sret %agg.result, i32 %dummy) nounwind {
entry:
	getelementptr %struct.sret0* %agg.result, i32 0, i32 0		; <i32*>:0 [#uses=1]
	store i32 %dummy, i32* %0, align 4
	getelementptr %struct.sret0* %agg.result, i32 0, i32 1		; <i32*>:1 [#uses=1]
	store i32 %dummy, i32* %1, align 4
	getelementptr %struct.sret0* %agg.result, i32 0, i32 2		; <i32*>:2 [#uses=1]
	store i32 %dummy, i32* %2, align 4
	ret void
}

