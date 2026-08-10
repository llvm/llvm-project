; This is a basic correctness check for constant propagation.  It tests the
; basic logic operations.


; RUN: opt < %s -passes=sccp -S | not grep and
; RUN: opt < %s -passes=sccp -S | not grep trunc
; RUN: opt < %s -passes=sccp -S | grep "ret i100 -1"

define i100 @test(i133 %A) {
        %B = and i133 0, %A
        %C = icmp sgt i133 %B, 0
	br i1 %C, label %BB1, label %BB2
BB1:
        %t3 = xor i133 %B, -1
        %t4 = trunc i133 %t3 to i100
	br label %BB3
BB2:
        %f1 = or i133 -1, %A
        %f2 = lshr i133 %f1, 33
        %f3 = trunc i133 %f2 to i100
	br label %BB3
BB3:
	%Ret = phi i100 [%t4, %BB1], [%f3, %BB2]
	ret i100 %Ret
}
