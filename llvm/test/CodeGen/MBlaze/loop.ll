; Test some complicated looping constructs to ensure that they
; compile successfully and that some sort of branching is used
; in the resulting code.
;
; RUN: llc < %s -march=mblaze -mattr=+mul,+fpu,+barrel | FileCheck %s

declare i32 @printf(i8*, ...)
@MSG = internal constant [19 x i8] c"Message: %d %d %d\0A\00"

define i32 @loop(i32 %a, i32 %b)
{
    ; CHECK:        loop:
entry:
    br label %loop_outer

loop_outer:
    %outer.0 = phi i32 [ 0, %entry ], [ %outer.2, %loop_outer_finish ]
    br label %loop_inner

loop_inner:
    %inner.0 = phi i32 [ %a, %loop_outer ], [ %inner.3, %loop_inner_finish ]
    %inner.1 = phi i32 [ %b, %loop_outer ], [ %inner.4, %loop_inner_finish ]
    %inner.2 = phi i32 [  0, %loop_outer ], [ %inner.5, %loop_inner_finish ]
    %inner.3 = add i32 %inner.0, %inner.1
    %inner.4 = mul i32 %inner.2, 11
    br label %loop_inner_finish

loop_inner_finish:
    %inner.5 = add i32 %inner.2, 1
    call i32 (i8*,...)* @printf( i8* getelementptr([19 x i8]* @MSG,i32 0,i32 0),
                                 i32 %inner.0, i32 %inner.1, i32 %inner.2 )

    %inner.6 = icmp eq i32 %inner.5, 100
    ; CHECK:        cmp [[REG:r[0-9]*]]

    br i1 %inner.6, label %loop_inner, label %loop_outer_finish
    ; CHECK:        {{beqid|bneid}} [[REG]]

loop_outer_finish:
    %outer.1 = add i32 %outer.0, 1
    %outer.2 = urem i32 %outer.1, 1500
    br label %loop_outer
    ; CHECK:        br
}
