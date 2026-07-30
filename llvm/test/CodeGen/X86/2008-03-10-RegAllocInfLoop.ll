; RUN: llc < %s -mtriple=i386-pc-linux-gnu -relocation-model=pic -frame-pointer=all
; PR2134

declare fastcc ptr @w_addchar(ptr, ptr, ptr, i8 signext ) nounwind 

define x86_stdcallcc i32 @parse_backslash(ptr inreg  %word, ptr inreg  %word_length, ptr inreg  %max_length) nounwind  {
entry:
	%tmp6 = load i8, ptr null, align 1		; <i8> [#uses=1]
	br label %bb13
bb13:		; preds = %entry
	%tmp26 = call fastcc ptr @w_addchar( ptr null, ptr %word_length, ptr %max_length, i8 signext  %tmp6 ) nounwind 		; <ptr> [#uses=1]
	store ptr %tmp26, ptr %word, align 4
	ret i32 0
}
