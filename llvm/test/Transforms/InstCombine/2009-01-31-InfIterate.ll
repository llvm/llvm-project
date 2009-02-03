; RUN: llvm-as < %s | opt -instcombine | llvm-dis
; PR3452
define i128 @test(i64 %A, i64 %B, i1 %C, i128 %Z, i128 %Y, i64* %P, i64* %Q) {
entry:
	%tmp2 = trunc i128 %Z to i64
	%tmp4 = trunc i128 %Y to i64
	store i64 %tmp2, i64* %P
	store i64 %tmp4, i64* %Q
	%x = sub i64 %tmp2, %tmp4
	%c = sub i64 %tmp2, %tmp4
	%tmp137 = zext i1 %C to i64
	%tmp138 = sub i64 %c, %tmp137
	br label %T

T:
	%G = phi i64 [%tmp138, %entry], [%tmp2, %Fal]
	%F = zext i64 %G to i128
	ret i128 %F

Fal:
	br label %T
}
