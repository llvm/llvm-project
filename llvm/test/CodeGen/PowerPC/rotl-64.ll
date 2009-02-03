; RUN: llvm-as < %s | llc -march=ppc64 | grep rldicl
; RUN: llvm-as < %s | llc -march=ppc64 | grep rldcl
; PR1613

define i64 @t1(i64 %A) {
	%tmp1 = lshr i64 %A, 57
        %tmp2 = shl i64 %A, 7
        %tmp3 = or i64 %tmp1, %tmp2
	ret i64 %tmp3
}

define i64 @t2(i64 %A, i8 zeroext %Amt) {
	%Amt1 = zext i8 %Amt to i64
	%tmp1 = lshr i64 %A, %Amt1
        %Amt2  = sub i8 64, %Amt
	%Amt3 = zext i8 %Amt2 to i64
        %tmp2 = shl i64 %A, %Amt3
        %tmp3 = or i64 %tmp1, %tmp2
	ret i64 %tmp3
}
