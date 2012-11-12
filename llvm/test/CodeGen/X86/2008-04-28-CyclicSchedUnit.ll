; RUN: llc < %s -march=x86

define i64 @t(i64 %maxIdleDuration) nounwind  {
	call void asm sideeffect "wrmsr", "{cx},A,~{dirflag},~{fpsr},~{flags}"( i32 416, i64 0 ) nounwind 
	unreachable
}
