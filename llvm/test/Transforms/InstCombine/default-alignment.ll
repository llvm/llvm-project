; RUN: opt -verify -instcombine < %s
%Foo = type <{ i8, x86_fp80 }>

define i8 @t(%Foo* %arg) {
entry:
  %0 = getelementptr %Foo, %Foo* %arg, i32 0, i32 0
  %1 = load i8, i8* %0, align 1
  ret i8 %1
}

