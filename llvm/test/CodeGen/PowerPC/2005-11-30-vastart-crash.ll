; RUN: llc -verify-machineinstrs < %s

target datalayout = "E-p:32:32"
target triple = "powerpc-unknown-linux-gnu"

define void @bar(i32 %G, i32 %E, i32 %F, i32 %A, i32 %B, i32 %C, i32 %D, ptr %fmt, ...) {
        %ap = alloca ptr                ; <ptr> [#uses=2]
        call void @llvm.va_start( ptr %ap )
        %tmp.1 = load ptr, ptr %ap          ; <ptr> [#uses=1]
        %tmp.0 = call double @foo( ptr %tmp.1 )         ; <double> [#uses=0]
        ret void
}

declare void @llvm.va_start(ptr)

declare double @foo(ptr)

