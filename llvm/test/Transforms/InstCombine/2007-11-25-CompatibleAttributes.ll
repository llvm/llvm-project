; RUN: opt < %s -passes=instcombine -S | not grep bitcast
; PR1716

@.str = internal constant [4 x i8] c"%d\0A\00"		; <ptr> [#uses=1]

define i32 @main(i32 %argc, ptr %argv) {
entry:
	%tmp32 = tail call i32 (ptr  , ...) @printf( ptr @.str  , i32 0 ) nounwind 		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @printf(ptr, ...) nounwind 
