; RUN: llvm-as %s -o - | llvm-dis > %t1.ll; 
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

declare void @llvm.va_start(i8** %ap)
declare void @llvm.va_copy(i8** %aq, i8** %ap)
declare void @llvm.va_end(i8** %ap)

define i33 @test(i33 %X, ...) {
        %ap = alloca i8*
	call void @llvm.va_start(i8** %ap)
	%tmp = va_arg i8** %ap, i33 

        %aq = alloca i8*
	call void @llvm.va_copy(i8** %aq, i8** %ap)
	call void @llvm.va_end(i8** %aq)
	
	call void @llvm.va_end(i8** %ap)
	ret i33 %tmp
}
