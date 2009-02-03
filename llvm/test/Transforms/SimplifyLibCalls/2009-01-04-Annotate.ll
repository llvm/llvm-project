; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis > %t
; RUN: grep noalias %t | count 2
; RUN: grep nocapture %t | count 3
; RUN: grep nounwind %t | count 3
; RUN: grep readonly %t | count 1

declare i8* @fopen(i8*, i8*)
declare i8 @strlen(i8*)
declare i32* @realloc(i32*, i32)

; Test deliberately wrong declaration
declare i32 @strcpy(...)
