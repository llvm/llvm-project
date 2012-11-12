; Ensure that the select instruction is supported and is lowered to 
; some sort of branch instruction.
;
; RUN: llc < %s -march=mblaze | FileCheck %s

define i32 @testsel(i32 %a, i32 %b)
{
    ; CHECK:        testsel:
    %tmp.1 = icmp eq i32 %a, %b
    ; CHECK:        cmp
    %tmp.2 = select i1 %tmp.1, i32 %a, i32 %b
    ; CHECK:        {{bne|beq}}
    ret i32 %tmp.2
    ; CHECK:        rtsd
}
