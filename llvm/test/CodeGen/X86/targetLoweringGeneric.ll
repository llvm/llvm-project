; RUN: llc -mtriple=i386-apple-darwin9 -mcpu=corei7 -fast-isel=false -O0 < %s | FileCheck %s

; Gather non-machine specific tests for the transformations in
; CodeGen/SelectionDAG/TargetLowering.  Currently, these
; can't be tested easily by checking the SDNodes that are
; the data structures that these transformations act on.
; Therefore, use X86 assembler output to check against.

; rdar://11195364 A problem with the transformation:
;  If all of the demanded bits on one side are known, and all of the set
;  bits on that side are also known to be set on the other side, turn this
;  into an AND, as we know the bits will be cleared.
; The known set (one) bits for the arguments %xor1 are not the same, so the
; transformation should not occur
define void @foo(i32 %i32In1, i32 %i32In2, i32 %i32In3, i32 %i32In4, 
                 i32 %i32In5, i32 %i32In6, ptr %i32StarOut, i1 %i1In1, 
                 ptr %i32SelOut) nounwind {
    %and3 = and i32 %i32In1, 1362779777
    %or2 = or i32 %i32In2, %i32In3
    %and2 = and i32 %or2, 1362779777
    %xor3 = xor i32 %and3, %and2
    ; CHECK: shll
    %shl1 = shl i32 %xor3, %i32In4
    %sub1 = sub i32 %or2, %shl1
    %add1 = add i32 %sub1, %i32In5
    %and1 = and i32 %add1, 1
    %xor2 = xor i32 %and1, 1
    %or1 = or i32 %xor2, 364806994 ;0x15BE8352
    ; CHECK-NOT: andl $96239955
    %xor1 = xor i32 %or1, 268567040 ;0x10020200
    ; force an output so not DCE'd
    store i32 %xor1, ptr %i32StarOut
    ; force not fast isel by using a select
    %i32SelVal = select i1 %i1In1, i32 %i32In1, i32 %xor1
    store i32 %i32SelVal, ptr %i32SelOut
    ; CHECK: ret
    ret void
}
