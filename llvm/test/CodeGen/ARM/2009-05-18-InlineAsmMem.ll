; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s
; PR4091

define void @foo(i32 %i, ptr %p) nounwind {
;CHECK: swp r2, r0, [r1]
	%asmtmp = call i32 asm sideeffect "swp $0, $2, $3", "=&r,=*m,r,*m,~{memory}"(ptr elementtype(i32) %p, i32 %i, ptr elementtype(i32) %p) nounwind
	ret void
}
