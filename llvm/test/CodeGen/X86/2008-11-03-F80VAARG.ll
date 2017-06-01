; RUN: llc < %s -march=x86 -o - | FileCheck %s

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_copy(i8*, i8*) nounwind

declare void @llvm.va_end(i8*) nounwind

; CHECK-LABEL: test:
; CHECK-NOT: 10
define x86_fp80 @test(...) nounwind {
	%ap = alloca i8*		; <i8**> [#uses=3]
	%v1 = bitcast i8** %ap to i8*		; <i8*> [#uses=1]
	call void @llvm.va_start(i8* %v1)
	%t1 = va_arg i8** %ap, x86_fp80		; <x86_fp80> [#uses=1]
	%t2 = va_arg i8** %ap, x86_fp80		; <x86_fp80> [#uses=1]
	%t = fadd x86_fp80 %t1, %t2		; <x86_fp80> [#uses=1]
	ret x86_fp80 %t
}
