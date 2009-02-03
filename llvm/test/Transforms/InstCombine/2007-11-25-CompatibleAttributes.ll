; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep bitcast
; PR1716

@.str = internal constant [4 x i8] c"%d\0A\00"		; <[4 x i8]*> [#uses=1]

define i32 @main(i32 %argc, i8** %argv) {
entry:
	%tmp32 = tail call i32 (i8* noalias , ...) nounwind * bitcast (i32 (i8*, ...) nounwind * @printf to i32 (i8* noalias , ...) nounwind *)( i8* getelementptr ([4 x i8]* @.str, i32 0, i32 0) noalias , i32 0 ) nounwind 		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @printf(i8*, ...) nounwind 
