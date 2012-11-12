; RUN: llc -march=mips < %s | FileCheck %s

define i32 @twoalloca(i32 %size) nounwind {
entry:
; CHECK: subu ${{[0-9]+}}, $sp
; CHECK: subu ${{[0-9]+}}, $sp
  alloca i8, i32 %size    ; <i8*>:0 [#uses=1]
  alloca i8, i32 %size    ; <i8*>:1 [#uses=1]
  call i32 @foo( i8* %0 ) nounwind    ; <i32>:2 [#uses=1]
  call i32 @foo( i8* %1 ) nounwind    ; <i32>:3 [#uses=1]
  add i32 %3, %2    ; <i32>:4 [#uses=1]
  ret i32 %4
}

declare i32 @foo(i8*)
