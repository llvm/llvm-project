; RUN: llvm-as < %s | llc -march=c

; Test const pointer refs & forward references

@t3 = global i32* @t1           ; <i32**> [#uses=0]
@t1 = global i32 4              ; <i32*> [#uses=1]

