; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    not grep memmove.i32
; Instcombine was trying to turn this into a memmove.i32

target datalayout = "e-p:64:64"
target triple = "alphaev67-unknown-linux-gnu"
@str10 = internal constant [1 x i8] zeroinitializer             ; <[1 x i8]*> [#uses=1]

define void @do_join(i8* %b) {
entry:
        call void @llvm.memmove.i64( i8* %b, i8* getelementptr ([1 x i8]* @str10, i32 0, i64 0), i64 1, i32 1 )
        ret void
}

declare void @llvm.memmove.i64(i8*, i8*, i64, i32)

