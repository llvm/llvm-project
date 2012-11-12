; RUN: llvm-as %s -disable-output
@G = constant <3 x i64> ptrtoint (<3 x i8*> <i8* null, i8* null, i8* null> to <3 x i64>)

@G1 = global i8 zeroinitializer
@g = constant <2 x i8*> getelementptr (<2 x i8*> <i8* @G1, i8* @G1>, <2 x i32> <i32 0, i32 0>)

@t = constant <2 x i1> icmp ((<2 x i32> ptrtoint (<2 x i8*> zeroinitializer to <2 x i32>), <2 x i32> zeroinitializer )

