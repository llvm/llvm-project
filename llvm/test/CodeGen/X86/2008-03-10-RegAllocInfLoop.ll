; RUN: llc < %s -mtriple=i386-pc-linux-gnu -relocation-model=pic -disable-fp-elim
; PR2134

declare fastcc i8* @w_addchar(i8*, i32*, i32*, i8 signext ) nounwind 

define x86_stdcallcc i32 @parse_backslash(i8** inreg  %word, i32* inreg  %word_length, i32* inreg  %max_length) nounwind  {
entry:
	%tmp6 = load i8* null, align 1		; <i8> [#uses=1]
	br label %bb13
bb13:		; preds = %entry
	%tmp26 = call fastcc i8* @w_addchar( i8* null, i32* %word_length, i32* %max_length, i8 signext  %tmp6 ) nounwind 		; <i8*> [#uses=1]
	store i8* %tmp26, i8** %word, align 4
	ret i32 0
}
