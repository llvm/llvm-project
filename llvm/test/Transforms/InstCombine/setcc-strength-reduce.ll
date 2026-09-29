; This test ensures that "strength reduction" of conditional expressions are
; working.  Basically this boils down to converting setlt,gt,le,ge instructions
; into equivalent setne,eq instructions.
;
; RUN: opt < %s -passes=instcombine -S | \
; RUN:    grep -v "icmp eq" | grep -v "icmp ne" | not grep icmp
; END.

define i1 @test1(i32 %A) {
        ; setne %A, 0
        %B = icmp uge i32 %A, 1         ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test2(i32 %A) {
       ; setne %A, 0
        %B = icmp ugt i32 %A, 0         ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test3(i8 %A) {
        ; setne %A, -128
        %B = icmp sge i8 %A, -127               ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test4(i8 %A) {
        ; setne %A, 127 
        %B = icmp sle i8 %A, 126                ; <i1> [#uses=1]
        ret i1 %B
}

define i1 @test5(i8 %A) {
        ; setne %A, 127
        %B = icmp slt i8 %A, 127                ; <i1> [#uses=1]
        ret i1 %B
}
