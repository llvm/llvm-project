; RUN: llc < %s -march=x86 | not grep movz
; PR2835

@g_407 = internal global i32 0		; <i32*> [#uses=1]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32 ()* @main to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i32 @main() nounwind {
entry:
	%0 = load volatile i32* @g_407, align 4		; <i32> [#uses=1]
	%1 = trunc i32 %0 to i8		; <i8> [#uses=1]
	%2 = tail call i32 @func_45(i8 zeroext %1) nounwind		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @func_45(i8 zeroext) nounwind
