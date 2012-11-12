; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | FileCheck %s -check-prefix=LINUX

; PR4639
@G1 = internal thread_local global i32 0		; <i32*> [#uses=1]
; LINUX: .section	.tbss,"awT",@nobits
; LINUX: G1:


define i32* @foo() nounwind readnone {
entry:
	ret i32* @G1
}


