; RUN: %lli -jit-kind=mcjit %s > /dev/null

;
; Regression Test: EnvironmentTest.ll
;
; Description:
;	This is a regression test that verifies that the JIT passes the
;	environment to the main() function.
;


declare i32 @strlen(ptr)

define i32 @main(i32 %argc.1, ptr %argv.1, ptr %envp.1) {
	%tmp.2 = load ptr, ptr %envp.1		; <ptr> [#uses=1]
	%tmp.3 = call i32 @strlen( ptr %tmp.2 )		; <i32> [#uses=1]
	%T = icmp eq i32 %tmp.3, 0		; <i1> [#uses=1]
	%R = zext i1 %T to i32		; <i32> [#uses=1]
	ret i32 %R
}

