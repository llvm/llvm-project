; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32--
; <rdar://problem/6020042>

define i32 @bork() nounwind  {
entry:
	br i1 true, label %bb1, label %bb3

bb1:
	%tmp1 = load i8, i8* null, align 1
	%tmp2 = icmp eq i8 %tmp1, 0
	br label %bb2

bb2:
	%val1 = phi i32 [ 0, %bb1 ], [ %val2, %bb2 ]
	%val2 = select i1 %tmp2, i32 -1, i32 %val1
	switch i32 %val2, label %bb2 [
		 i32 -1, label %bb3
		 i32 0, label %bb1
		 i32 1, label %bb3
		 i32 2, label %bb1
	]

bb3:
	ret i32 -1
}
