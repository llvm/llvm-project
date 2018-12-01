;RUN: opt -mem2reg -S < %s | FileCheck %s

declare i1 @cond()

define i32 @foo() {
Entry:
    %val = alloca i32
    %c1 = call i1 @cond()
    br i1 %c1, label %Store1, label %Store2
Block1:
    br label %Join
Block2:
    br label %Join
Block3:
    br label %Join
Block4:
    br label %Join
Block5:
    br label %Join
Store1:
    store i32 1, i32* %val
    br label %Join
Block6:
    br label %Join
Block7:
    br label %Join
Block8:
    br label %Join
Block9:
    br label %Join
Block10:
    br label %Join
Store2:
    store i32 2, i32* %val
    br label %Join
Block11:
    br label %Join
Block12:
    br label %Join
Block13:
    br label %Join
Block14:
    br label %Join
Block15:
    br label %Join
Block16:
    br label %Join
Join:
; Phi inserted here should have operands appended deterministically
; CHECK: %val.0 = phi i32 [ 1, %Store1 ], [ 2, %Store2 ], [ undef, %Block1 ], [ undef, %Block2 ], [ undef, %Block3 ], [ undef, %Block4 ], [ undef, %Block5 ], [ undef, %Block6 ], [ undef, %Block7 ], [ undef, %Block8 ], [ undef, %Block9 ], [ undef, %Block10 ], [ undef, %Block11 ], [ undef, %Block12 ], [ undef, %Block13 ], [ undef, %Block14 ], [ undef, %Block15 ], [ undef, %Block16 ]
    %result = load i32, i32* %val
    ret i32 %result
}
