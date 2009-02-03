; Test that this transform works:
; udiv X, (Select Cond, C1, C2) --> Select Cond, (shr X, C1), (shr X, C2)
;
; RUN: llvm-as < %s | opt -instcombine | llvm-dis -f -o %t
; RUN:   not grep select %t
; RUN:   grep lshr %t | count 2
; RUN:   not grep udiv %t

define i64 @test(i64 %X, i1 %Cond ) {
entry:
        %divisor1 = select i1 %Cond, i64 16, i64 8
        %quotient1 = udiv i64 %X, %divisor1
        %divisor2 = select i1 %Cond, i64 8, i64 0
        %quotient2 = udiv i64 %X, %divisor2
        %sum = add i64 %quotient1, %quotient2
        ret i64 %sum
}
