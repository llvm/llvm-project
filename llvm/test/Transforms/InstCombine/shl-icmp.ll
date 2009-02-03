; RUN: llvm-as < %s | opt -instcombine -stats -disable-output |& \
; RUN:   grep {Number of insts combined} | grep 5

define i8 @t1(i8 zeroext %x, i8 zeroext %y) zeroext nounwind {
entry:
	%tmp1 = lshr i8 %x, 7
	%cond1 = icmp ne i8 %tmp1, 0
	br i1 %cond1, label %bb1, label %bb2

bb1:
	ret i8 %tmp1

bb2:
        %tmp2 = add i8 %tmp1, %y
	ret i8 %tmp2
}

define i8 @t2(i8 zeroext %x) zeroext nounwind {
entry:
	%tmp1 = lshr i8 %x, 7
	%cond1 = icmp ne i8 %tmp1, 0
	br i1 %cond1, label %bb1, label %bb2

bb1:
	ret i8 0

bb2:
	ret i8 1
}
