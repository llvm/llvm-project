; RUN: llc < %s

target datalayout = "E-p:32:32"
target triple = "powerpc-apple-darwin8.2.0"

define void @bar(i32 %G, i32 %E, i32 %F, i32 %A, i32 %B, i32 %C, i32 %D, i8* %fmt, ...) {
        %ap = alloca i8*                ; <i8**> [#uses=2]
        %va.upgrd.1 = bitcast i8** %ap to i8*           ; <i8*> [#uses=1]
        call void @llvm.va_start( i8* %va.upgrd.1 )
        %tmp.1 = load i8** %ap          ; <i8*> [#uses=1]
        %tmp.0 = call double @foo( i8* %tmp.1 )         ; <double> [#uses=0]
        ret void
}

declare void @llvm.va_start(i8*)

declare double @foo(i8*)

