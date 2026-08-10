; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target triple = "powerpc-unknown-linux-gnu"
@str = internal constant [18 x i8] c"hello world!, %d\0A\00"            ; <ptr> [#uses=1]


define i32 @main() {
entry:
; CHECK: main:
; CHECK: mflr
; CHECK-NOT: mflr
; CHECK: mtlr
        %tmp = tail call i32 (ptr, ...) @printf( ptr @str )                ; <i32> [#uses=0]
        ret i32 0
}

declare i32 @printf(ptr, ...)
