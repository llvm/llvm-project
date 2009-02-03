; RUN: llvm-as < %s | llc -march=c

@array = constant [2 x i32] [ i32 12, i32 52 ]          ; <[2 x i32]*> [#uses=1]
@arrayPtr = global i32* getelementptr ([2 x i32]* @array, i64 0, i64 0)         ; <i32**> [#uses=0]

