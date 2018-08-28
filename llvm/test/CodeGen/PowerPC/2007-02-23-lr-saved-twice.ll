; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target triple = "powerpc-unknown-linux-gnu"
@str = internal constant [18 x i8] c"hello world!, %d\0A\00"            ; <[18 x i8]*> [#uses=1]


define i32 @main() {
entry:
; CHECK: main:
; CHECK: mflr
; CHECK-NOT: mflr
; CHECK: mtlr
        %tmp = tail call i32 (i8*, ...) @printf( i8* getelementptr ([18 x i8], [18 x i8]* @str, i32 0, i32 0) )                ; <i32> [#uses=0]
        ret i32 0
}

declare i32 @printf(i8*, ...)
