; RUN: llvm-as < %s | opt -scalarrepl -disable-output

target datalayout = "E-p:32:32"
	%struct.rtx_def = type { [2 x i8], i32, [1 x %union.rtunion_def] }
	%union.rtunion_def = type { i32 }

define void @find_reloads() {
entry:
	%c_addr.i = alloca i8		; <i8*> [#uses=1]
	switch i32 0, label %return [
		 i32 36, label %label.7
		 i32 34, label %label.7
		 i32 41, label %label.5
	]
label.5:		; preds = %entry
	ret void
label.7:		; preds = %entry, %entry
	br i1 false, label %then.4, label %switchexit.0
then.4:		; preds = %label.7
	%tmp.0.i = bitcast i8* %c_addr.i to i32*		; <i32*> [#uses=1]
	store i32 44, i32* %tmp.0.i
	ret void
switchexit.0:		; preds = %label.7
	ret void
return:		; preds = %entry
	ret void
}

