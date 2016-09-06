; An integer truncation to i1 should be done with an and instruction to make
; sure only the LSBit survives. Test that this is the case both for a returned
; value and as the operand of a branch.
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s

define zeroext i1 @test1(i32 %X)  nounwind {
    %Y = trunc i32 %X to i1
    ret i1 %Y
}
; CHECK-LABEL: test1:
; CHECK: andb $1, %al

define i1 @test2(i32 %val, i32 %mask) nounwind {
entry:
    %shifted = ashr i32 %val, %mask
    %anded = and i32 %shifted, 1
    %trunced = trunc i32 %anded to i1
    br i1 %trunced, label %ret_true, label %ret_false
ret_true:
    ret i1 true
ret_false:
    ret i1 false
}
; CHECK-LABEL: test2:
; CHECK: btl

define i32 @test3(i8* %ptr) nounwind {
    %val = load i8, i8* %ptr
    %tmp = trunc i8 %val to i1
    br i1 %tmp, label %cond_true, label %cond_false
cond_true:
    ret i32 21
cond_false:
    ret i32 42
}
; CHECK-LABEL: test3:
; CHECK: testb $1, (%eax)

define i32 @test4(i8* %ptr) nounwind {
    %tmp = ptrtoint i8* %ptr to i1
    br i1 %tmp, label %cond_true, label %cond_false
cond_true:
    ret i32 21
cond_false:
    ret i32 42
}
; CHECK-LABEL: test4:
; CHECK: testb $1, 4(%esp)

define i32 @test5(double %d) nounwind {
    %tmp = fptosi double %d to i1
    br i1 %tmp, label %cond_true, label %cond_false
cond_true:
    ret i32 21
cond_false:
    ret i32 42
}
; CHECK-LABEL: test5:
; CHECK: testb $1
