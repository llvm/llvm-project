; RUN: opt < %s -passes=instcombine -S | grep "store i8" | count 2

define i32 @a(ptr %s) nounwind  {
entry:
	store i8 0, ptr %s, align 1 ; This store cannot be eliminated!
	%tmp3 = call i32 @strlen( ptr %s ) nounwind readonly
	%tmp5 = icmp ne i32 %tmp3, 0
	br i1 %tmp5, label %bb, label %bb8

bb:		; preds = %entry
	store i8 0, ptr %s, align 1
	br label %bb8

bb8:
	ret i32 %tmp3
}

declare i32 @strlen(ptr) nounwind readonly 

