; RUN: llc -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

@a = global i32 1
@b = global i32 1, !exclude !0
@c = global i32 1, section "aaa"
; CHECK-DAG: .type	c,@object
; CHECK-DAG: 	.section	aaa,"aw",@progbits
@d = global i32 1, section "bbb", !exclude !0
; CHECK-DAG: .type	d,@object
; CHECK-DAG: 	.section	bbb,"e",@progbits
@e = global i32 1, section "bbb", !exclude !0
; CHECK-DAG: .type	e,@object
@f = global i32 1, section "ccc", !exclude !0
@g = global i32 1, section "ccc"
; CHECK-DAG:	.type	f,@object
; CHECK-DAG:	.section	ccc,"e",@progbits

!0 = !{}
