; RUN: opt < %s -instcombine -S | FileCheck %s

define void @test(<4 x i32> %v, i64 *%r1, i64 *%r2) {
;CHECK: %1 = extractelement <4 x i32> %v, i32 0
;CHECK: %2 = zext i32 %1 to i64
        %1 = zext <4 x i32> %v to <4 x i64>
        %2 = extractelement <4 x i64> %1, i32 0
        store i64 %2, i64 *%r1
        store i64 %2, i64 *%r2
        ret void
}

