; Ensure that indirect calls work and that they are lowered to some
; sort of branch and link instruction.
;
; RUN: llc < %s -march=mblaze -mattr=+mul,+fpu,+barrel | FileCheck %s

declare i32 @printf(i8*, ...)
@MSG = internal constant [13 x i8] c"Message: %d\0A\00"

@FUNS = private constant [5 x i32 (i32,i32)*]
    [ i32 (i32,i32)* @doadd,
      i32 (i32,i32)* @dosub,
      i32 (i32,i32)* @domul,
      i32 (i32,i32)* @dodiv,
      i32 (i32,i32)* @dorem ]

define i32 @doadd(i32 %a, i32 %b)
{
    ; CHECK:        doadd:
    %tmp.0 = add i32 %a, %b
    ret i32 %tmp.0
    ; CHECK:        rtsd
}

define i32 @dosub(i32 %a, i32 %b)
{
    ; CHECK:        dosub:
    %tmp.0 = sub i32 %a, %b
    ret i32 %tmp.0
    ; CHECK:        rtsd
}

define i32 @domul(i32 %a, i32 %b)
{
    ; CHECK:        domul:
    %tmp.0 = mul i32 %a, %b
    ret i32 %tmp.0
    ; CHECK:        rtsd
}

define i32 @dodiv(i32 %a, i32 %b)
{
    ; CHECK:        dodiv:
    %tmp.0 = sdiv i32 %a, %b
    ret i32 %tmp.0
    ; CHECK:        rtsd
}

define i32 @dorem(i32 %a, i32 %b)
{
    ; CHECK:        dorem:
    %tmp.0 = srem i32 %a, %b
    ret i32 %tmp.0
    ; CHECK:        rtsd
}

define i32 @callind(i32 %a, i32 %b)
{
    ; CHECK:        callind:
entry:
    br label %loop

loop:
    %tmp.0 = phi i32 [ 0, %entry ], [ %tmp.3, %loop ]
    %dst.0 = getelementptr [5 x i32 (i32,i32)*]* @FUNS, i32 0, i32 %tmp.0
    %dst.1 = load i32 (i32,i32)** %dst.0
    %tmp.1 = call i32 %dst.1(i32 %a, i32 %b)
    ; CHECK-NOT:    brli
    ; CHECK-NOT:    brlai
    ; CHECK:        brl

    call i32 (i8*,...)* @printf( i8* getelementptr([13 x i8]* @MSG,i32 0,i32 0),
                                 i32 %tmp.1)
    ; CHECK:        brl

    %tmp.2 = add i32 %tmp.0, 1
    %tmp.3 = urem i32 %tmp.2, 5

    br label %loop
    ; CHECK:        br
}
