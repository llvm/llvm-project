; RUN: llc < %s -march=x86 -relocation-model=static | FileCheck %s
; PR882

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin9.0.0d2"
@GV = weak global i32 0		; <i32*> [#uses=2]
@str = external global [12 x i8]		; <[12 x i8]*> [#uses=1]

define void @foo() {
; CHECK: foo:
; CHECK-NOT: ret
; CHECK: test1 $_GV
; CHECK-NOT: ret
; CHECK: test2 _GV
; CHECK: ret

	tail call void asm sideeffect "test1 $0", "i,~{dirflag},~{fpsr},~{flags}"( i32* @GV )
	tail call void asm sideeffect "test2 ${0:c}", "i,~{dirflag},~{fpsr},~{flags}"( i32* @GV )
	ret void
}

define void @unknown_bootoption() {
entry:
	call void asm sideeffect "ud2\0A\09.word ${0:c}\0A\09.long ${1:c}\0A", "i,i,~{dirflag},~{fpsr},~{flags}"( i32 235, i8* getelementptr ([12 x i8]* @str, i32 0, i64 0) )
	ret void
}
