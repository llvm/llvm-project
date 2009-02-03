; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Demonstrate all of the variable argument handling intrinsic functions plus 
; the va_arg instruction.

declare void @llvm.va_start(i8*)

declare void @llvm.va_copy(i8*, i8*)

declare void @llvm.va_end(i8*)

define i32 @test(i32 %X, ...) {
        %ap = alloca i8*                ; <i8**> [#uses=4]
        %va.upgrd.1 = bitcast i8** %ap to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_start( i8* %va.upgrd.1 )
        %tmp = va_arg i8** %ap, i32             ; <i32> [#uses=1]
        %aq = alloca i8*                ; <i8**> [#uses=2]
        %va0.upgrd.2 = bitcast i8** %aq to i8*          ; <i8*> [#uses=1]
        %va1.upgrd.3 = bitcast i8** %ap to i8*          ; <i8*> [#uses=1]
        call void @llvm.va_copy( i8* %va0.upgrd.2, i8* %va1.upgrd.3 )
        %va.upgrd.4 = bitcast i8** %aq to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_end( i8* %va.upgrd.4 )
        %va.upgrd.5 = bitcast i8** %ap to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_end( i8* %va.upgrd.5 )
        ret i32 %tmp
}

