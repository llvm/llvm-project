; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {store i8} | count 2

define i32 @a(i8* %s) nounwind  {
entry:
	store i8 0, i8* %s, align 1 ; This store cannot be eliminated!
	%tmp3 = call i32 @strlen( i8* %s ) nounwind readonly
	%tmp5 = icmp ne i32 %tmp3, 0
	br i1 %tmp5, label %bb, label %bb8

bb:		; preds = %entry
	store i8 0, i8* %s, align 1
	br label %bb8

bb8:
	ret i32 %tmp3
}

declare i32 @strlen(i8*) nounwind readonly 

