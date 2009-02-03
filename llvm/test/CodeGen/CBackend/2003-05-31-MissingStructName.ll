; RUN: llvm-as < %s | llc -march=c

; The C backend was dying when there was no typename for a struct type!

declare i32 @test(i32, { [32 x i32] }*)
