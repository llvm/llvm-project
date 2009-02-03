; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

declare void @llvm.va_start(i8*)

declare void @llvm.va_copy(i8*, i8*)

declare void @llvm.va_end(i8*)

define i32 @test(i32 %X, ...) {
        ; Allocate two va_list items.  On this target, va_list is of type sbyte*
        %ap = alloca i8*                ; <i8**> [#uses=4]
        %aq = alloca i8*                ; <i8**> [#uses=2]

        ; Initialize variable argument processing
        %va.upgrd.1 = bitcast i8** %ap to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_start( i8* %va.upgrd.1 )

        ; Read a single integer argument
        %tmp = va_arg i8** %ap, i32             ; <i32> [#uses=1]

        ; Demonstrate usage of llvm.va_copy and llvm_va_end
        %apv = load i8** %ap            ; <i8*> [#uses=1]
        %va0.upgrd.2 = bitcast i8** %aq to i8*          ; <i8*> [#uses=1]
        %va1.upgrd.3 = bitcast i8* %apv to i8*          ; <i8*> [#uses=1]
        call void @llvm.va_copy( i8* %va0.upgrd.2, i8* %va1.upgrd.3 )
        %va.upgrd.4 = bitcast i8** %aq to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_end( i8* %va.upgrd.4 )

        ; Stop processing of arguments.
        %va.upgrd.5 = bitcast i8** %ap to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_end( i8* %va.upgrd.5 )
        ret i32 %tmp
}
