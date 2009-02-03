; RUN: llvm-as < %s | opt -globalsmodref-aa -licm -disable-output

@PL_regcomp_parse = internal global i8* null		; <i8**> [#uses=2]

define void @test() {
	br label %Outer
Outer:		; preds = %Next, %0
	br label %Inner
Inner:		; preds = %Inner, %Outer
	%tmp.114.i.i.i = load i8** @PL_regcomp_parse		; <i8*> [#uses=1]
	%tmp.115.i.i.i = load i8* %tmp.114.i.i.i		; <i8> [#uses=0]
	store i8* null, i8** @PL_regcomp_parse
	br i1 false, label %Inner, label %Next
Next:		; preds = %Inner
	br i1 false, label %Outer, label %Exit
Exit:		; preds = %Next
	ret void
}

