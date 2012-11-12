; RUN: llc < %s -march=arm | FileCheck %s
; RUN: llc < %s -march=thumb | FileCheck %s
; PR4091

define void @foo(i32 %i, i32* %p) nounwind {
;CHECK: swp r2, r0, [r1]
	%asmtmp = call i32 asm sideeffect "swp $0, $2, $3", "=&r,=*m,r,*m,~{memory}"(i32* %p, i32 %i, i32* %p) nounwind
	ret void
}
