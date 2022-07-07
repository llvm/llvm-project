; RUN: llc -mtriple x86_64-win32-gnu < %s | FileCheck %s

@a = global i32 1
@b = global i32 1, !exclude !0
@c = global i32 1, section "aaa"
; CHECK-DAG: c
; CHECK-DAG: 	.section	aaa,"dw"
@d = global i32 1, section "bbb", !exclude !0
; CHECK-DAG: d
; CHECK-DAG: 	.section	bbb,"ynD"
@e = global i32 1, section "bbb", !exclude !0
; CHECK-DAG: e
@f = global i32 1, section "ccc", !exclude !0
@g = global i32 1, section "ccc"
; CHECK-DAG: f
; CHECK-DAG:	.section	ccc,"ynD"

!0 = !{}
