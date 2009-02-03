; RUN: llvm-as %s -o /dev/null -f

%T = type i32
@X = global i32* null           ; <i32**> [#uses=0]
@Y = global i32* null           ; <i32**> [#uses=0]

