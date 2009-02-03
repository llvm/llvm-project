; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

declare void @llvm.va_start(i8**)
declare void @llvm.va_copy(i8**, i8*)
declare void @llvm.va_end(i8**)

define i31 @test(i31 %X, ...) {
        ; Allocate two va_list items.  On this target, va_list is of type i8*
        %ap = alloca i8*             ; <i8**> [#uses=4]
        %aq = alloca i8*             ; <i8**> [#uses=2]

        ; Initialize variable argument processing
        call void @llvm.va_start(i8** %ap)

        ; Read a single integer argument
        %tmp = va_arg i8** %ap, i31           ; <i31> [#uses=1]

        ; Demonstrate usage of llvm.va_copy and llvm_va_end
        %apv = load i8** %ap         ; <i8*> [#uses=1]
        call void @llvm.va_copy(i8** %aq, i8* %apv)
        call void @llvm.va_end(i8** %aq)

        ; Stop processing of arguments.
        call void @llvm.va_end(i8** %ap)
        ret i31 %tmp

}
