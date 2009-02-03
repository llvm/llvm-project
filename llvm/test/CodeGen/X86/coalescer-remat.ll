; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin | grep xor | count 3

@val = internal global i64 0		; <i64*> [#uses=1]
@"\01LC" = internal constant [7 x i8] c"0x%lx\0A\00"		; <[7 x i8]*> [#uses=1]

define i32 @main() nounwind {
entry:
	%0 = tail call i64 @llvm.atomic.cmp.swap.i64.p0i64(i64* @val, i64 0, i64 1)		; <i64> [#uses=1]
	%1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([7 x i8]* @"\01LC", i32 0, i64 0), i64 %0) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare i64 @llvm.atomic.cmp.swap.i64.p0i64(i64*, i64, i64) nounwind

declare i32 @printf(i8*, ...) nounwind
