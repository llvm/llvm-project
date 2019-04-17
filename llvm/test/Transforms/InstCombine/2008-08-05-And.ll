; RUN: opt < %s -instcombine -S | not grep or
; PR2629

define void @f(i8* %x) nounwind  {
entry:
        br label %bb

bb:
	%g1 = getelementptr i8, i8* %x, i32 0
        %l1 = load i8, i8* %g1, align 1
	%s1 = sub i8 %l1, 6
	%c1 = icmp ugt i8 %s1, 2
	%s2 = sub i8 %l1, 10
        %c2 = icmp ugt i8 %s2, 2
        %a1 = and i1 %c1, %c2
	br i1 %a1, label %incompatible, label %okay

okay:
        ret void

incompatible:
        ret void
}
