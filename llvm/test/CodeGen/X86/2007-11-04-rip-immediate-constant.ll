; RUN: llc < %s -relocation-model=static | grep "foo str$"
; PR1761
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-pc-linux"
@str = internal constant [12 x i8] c"init/main.c\00"		; <[12 x i8]*> [#uses=1]

define i32 @unknown_bootoption() {
entry:
	tail call void asm sideeffect "foo ${0:c}\0A", "i,~{dirflag},~{fpsr},~{flags}"( i8* getelementptr ([12 x i8]* @str, i32 0, i64 0) )
	ret i32 undef
}
