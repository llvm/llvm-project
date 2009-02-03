; This is a basic sanity check for constant propogation.  It tests the basic 
; logic operations.


; RUN: llvm-as < %s | opt -sccp | llvm-dis | not grep and
; RUN: llvm-as < %s | opt -sccp | llvm-dis | not grep trunc
; RUN: llvm-as < %s | opt -sccp | llvm-dis | grep {ret i100 -1}

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
