; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | FileCheck %s -check-prefix=LINUX

; PR4639
@G1 = internal thread_local global i32 0		; <ptr> [#uses=1]
; LINUX: .section	.tbss,"awT",@nobits
; LINUX: G1:


define ptr @foo() nounwind readnone {
entry:
	ret ptr @G1
}


