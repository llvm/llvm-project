; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | not grep icmp
; END.
	
@X = external global i32                ; <i32*> [#uses=2]

define i1 @test1(i32 %A) {
        %B = icmp eq i32 %A, %A         ; <i1> [#uses=1]
        ; Never true
        %C = icmp eq i32* @X, null              ; <i1> [#uses=1]
        %D = and i1 %B, %C              ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test2(i32 %A) {
        %B = icmp ne i32 %A, %A         ; <i1> [#uses=1]
        ; Never false
        %C = icmp ne i32* @X, null              ; <i1> [#uses=1]
        %D = or i1 %B, %C               ; <i1> [#uses=1]
        ret i1 %D
}

define i1 @test3(i32 %A) {
        %B = icmp slt i32 %A, %A                ; <i1> [#uses=1]
        ret i1 %B
}


define i1 @test4(i32 %A) {
        %B = icmp sgt i32 %A, %A                ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test5(i32 %A) {
        %B = icmp sle i32 %A, %A                ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test6(i32 %A) {
        %B = icmp sge i32 %A, %A                ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test7(i32 %A) {
        ; true
        %B = icmp uge i32 %A, 0         ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test8(i32 %A) {
        ; false
        %B = icmp ult i32 %A, 0         ; <i1> [#uses=1]
        ret i1 %B
}

;; test operations on boolean values these should all be eliminated$a
define i1 @test9(i1 %A) {
        ; false
        %B = icmp ult i1 %A, false              ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test10(i1 %A) {
        ; false
        %B = icmp ugt i1 %A, true               ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test11(i1 %A) {
        ; true
        %B = icmp ule i1 %A, true               ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test12(i1 %A) {
        ; true
        %B = icmp uge i1 %A, false              ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test13(i1 %A, i1 %B) {
        ; A | ~B
        %C = icmp uge i1 %A, %B         ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test14(i1 %A, i1 %B) {
        ; ~(A ^ B)
        %C = icmp eq i1 %A, %B          ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test16(i32 %A) {
        %B = and i32 %A, 5              ; <i32> [#uses=1]
        ; Is never true
        %C = icmp eq i32 %B, 8          ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test17(i8 %A) {
        %B = or i8 %A, 1                ; <i8> [#uses=1]
        ; Always false
        %C = icmp eq i8 %B, 2           ; <i1> [#uses=1]
        ret i1 %C
}

define i1 @test18(i1 %C, i32 %a) {
entry:
        br i1 %C, label %endif, label %else

else:           ; preds = %entry
        br label %endif

endif:          ; preds = %else, %entry
        %b.0 = phi i32 [ 0, %entry ], [ 1, %else ]              ; <i32> [#uses=1]
        %tmp.4 = icmp slt i32 %b.0, 123         ; <i1> [#uses=1]
        ret i1 %tmp.4
}

define i1 @test19(i1 %A, i1 %B) {
        %a = zext i1 %A to i32          ; <i32> [#uses=1]
        %b = zext i1 %B to i32          ; <i32> [#uses=1]
        %C = icmp eq i32 %a, %b         ; <i1> [#uses=1]
        ret i1 %C
}

define i32 @test20(i32 %A) {
        %B = and i32 %A, 1              ; <i32> [#uses=1]
        %C = icmp ne i32 %B, 0          ; <i1> [#uses=1]
        %D = zext i1 %C to i32          ; <i32> [#uses=1]
        ret i32 %D
}

define i32 @test21(i32 %a) {
        %tmp.6 = and i32 %a, 4          ; <i32> [#uses=1]
        %not.tmp.7 = icmp ne i32 %tmp.6, 0              ; <i1> [#uses=1]
        %retval = zext i1 %not.tmp.7 to i32             ; <i32> [#uses=1]
        ret i32 %retval
}

define i1 @test22(i32 %A, i32 %X) {
        %B = and i32 %A, 100663295              ; <i32> [#uses=1]
        %C = icmp ult i32 %B, 268435456         ; <i1> [#uses=1]
        %Y = and i32 %X, 7              ; <i32> [#uses=1]
        %Z = icmp sgt i32 %Y, -1                ; <i1> [#uses=1]
        %R = or i1 %C, %Z               ; <i1> [#uses=1]
        ret i1 %R
}

define i32 @test23(i32 %a) {
        %tmp.1 = and i32 %a, 1          ; <i32> [#uses=1]
        %tmp.2 = icmp eq i32 %tmp.1, 0          ; <i1> [#uses=1]
        %tmp.3 = zext i1 %tmp.2 to i32          ; <i32> [#uses=1]
        ret i32 %tmp.3
}

define i32 @test24(i32 %a) {
        %tmp1 = and i32 %a, 4           ; <i32> [#uses=1]
        %tmp.1 = lshr i32 %tmp1, 2              ; <i32> [#uses=1]
        %tmp.2 = icmp eq i32 %tmp.1, 0          ; <i1> [#uses=1]
        %tmp.3 = zext i1 %tmp.2 to i32          ; <i32> [#uses=1]
        ret i32 %tmp.3
}

define i1 @test25(i32 %A) {
        %B = and i32 %A, 2              ; <i32> [#uses=1]
        %C = icmp ugt i32 %B, 2         ; <i1> [#uses=1]
        ret i1 %C
}

