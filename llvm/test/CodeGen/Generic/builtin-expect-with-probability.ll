; RUN: llc < %s

declare i32 @llvm.expect.with.probability(i32, i32, double)

define i32 @test1(i32 %val) nounwind {
    %expected = call i32 @llvm.expect.with.probability(i32 %val, i32 1, double 0.5)
    ret i32 %expected
}
