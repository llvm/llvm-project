; RUN: llvm-as < %s | llc -march=arm
; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnueabi | grep mov | count 1
; RUN: llvm-as < %s | llc -mtriple=arm-linux-gnu --disable-fp-elim | \
; RUN:   grep mov | count 3
; RUN: llvm-as < %s | llc -mtriple=arm-apple-darwin | grep mov | count 2

@str = internal constant [12 x i8] c"Hello World\00"

define i32 @main() {
	%tmp = call i32 @puts( i8* getelementptr ([12 x i8]* @str, i32 0, i64 0) )		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @puts(i8*)
