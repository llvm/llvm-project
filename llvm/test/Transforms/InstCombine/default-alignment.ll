; RUN: opt -passes=verify,instcombine < %s
%Foo = type <{ i8, x86_fp80 }>

define i8 @t(ptr %arg) {
entry:
  %0 = load i8, ptr %arg, align 1
  ret i8 %0
}

