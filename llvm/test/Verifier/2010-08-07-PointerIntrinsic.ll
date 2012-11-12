; RUN: not llvm-as < %s 2> %t
; RUN: grep "Broken module" %t
; PR7316

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"
target triple = "x86-unknown-unknown"
@aa = global [32 x i8] zeroinitializer, align 1
@bb = global [16 x i8] zeroinitializer, align 1
define void @x() nounwind {
L.0:
	%0 = getelementptr [32 x i8]* @aa, i32 0, i32 4
	%1 = bitcast i8* %0 to [16 x i8]*
	%2 = bitcast [16 x i8]* %1 to [0 x i8]*
	%3 = getelementptr [16 x i8]* @bb
	%4 = bitcast [16 x i8]* %3 to [0 x i8]*
	call void @llvm.memcpy.i32([0 x i8]* %2, [0 x i8]* %4, i32 16, i32 1)
	br label %return
return:
	ret void
}
declare void @llvm.memcpy.i32([0 x i8]*, [0 x i8]*, i32, i32) nounwind
