; Ensure that the select instruction is supported and is lowered to 
; some sort of branch instruction.
;
; RUN: llc < %s -march=mblaze -mattr=+mul,+fpu,+barrel | FileCheck %s

declare i32 @printf(i8*, ...)
@MSG = internal constant [13 x i8] c"Message: %d\0A\00"

@BLKS = private constant [5 x i8*]
    [ i8* blockaddress(@brind, %L1),
      i8* blockaddress(@brind, %L2),
      i8* blockaddress(@brind, %L3),
      i8* blockaddress(@brind, %L4),
      i8* blockaddress(@brind, %L5) ]

define i32 @brind(i32 %a, i32 %b)
{
    ; CHECK:        brind:
entry:
    br label %loop

loop:
    %tmp.0 = phi i32 [ 0, %entry ], [ %tmp.8, %finish ]
    %dst.0 = getelementptr [5 x i8*]* @BLKS, i32 0, i32 %tmp.0
    %dst.1 = load i8** %dst.0
    indirectbr i8* %dst.1, [ label %L1,
                             label %L2,
                             label %L3,
                             label %L4,
                             label %L5 ]
    ; CHECK:        brad {{r[0-9]*}}

L1:
    %tmp.1 = add i32 %a, %b
    br label %finish
    ; CHECK:        brid

L2:
    %tmp.2 = sub i32 %a, %b
    br label %finish
    ; CHECK:        brid

L3:
    %tmp.3 = mul i32 %a, %b
    br label %finish
    ; CHECK:        brid

L4:
    %tmp.4 = sdiv i32 %a, %b
    br label %finish
    ; CHECK:        brid

L5:
    %tmp.5 = srem i32 %a, %b
    br label %finish

finish:
    %tmp.6 = phi i32 [ %tmp.1, %L1 ],
                     [ %tmp.2, %L2 ],
                     [ %tmp.3, %L3 ],
                     [ %tmp.4, %L4 ],
                     [ %tmp.5, %L5 ]

    call i32 (i8*,...)* @printf( i8* getelementptr([13 x i8]* @MSG,i32 0,i32 0),
                                 i32 %tmp.6)

    %tmp.7 = add i32 %tmp.0, 1
    %tmp.8 = urem i32 %tmp.7, 5

    br label %loop
    ; CHECK:        brad {{r[0-9]*}}
}
