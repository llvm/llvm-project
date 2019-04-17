; RUN: opt < %s -ipsccp -S | grep -v "ret i512 undef" | \
; RUN:   grep "ret i8 2"

define internal i512 @test(i1 %B) {
	br i1 %B, label %BB1, label %BB2
BB1:
	%Val = add i512 0, 1
	br label %BB3
BB2:
	br label %BB3
BB3:
	%Ret = phi i512 [%Val, %BB1], [2, %BB2]
	ret i512 %Ret
}

define i8 @caller()
{
    %t1 = and i2 2, 1
    %t11 = trunc i2 %t1 to i1
    %t2 = call i512 @test(i1 %t11)
    %t3 = trunc i512 %t2 to i8
    ret i8 %t3
}

