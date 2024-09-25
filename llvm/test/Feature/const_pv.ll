; RUN: llvm-as %s -disable-output
@G = constant <3 x i64> ptrtoint (<3 x ptr> <ptr null, ptr null, ptr null> to <3 x i64>)

@G1 = global i8 zeroinitializer
@g = constant <2 x ptr> getelementptr (i8, <2 x ptr> <ptr @G1, ptr @G1>, <2 x i32> <i32 0, i32 0>)

