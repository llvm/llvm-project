; RUN: llc < %s -march=xcore | FileCheck %s

; CHECK-LABEL: bl_imm:
; CHECK: ldw [[R0:r[0-9]+]], cp
; CHECK: bla [[R0]]
define void @bl_imm() nounwind {
entry:
	tail call void inttoptr (i64 65536 to ptr)() nounwind
	ret void
}
