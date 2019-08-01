; RUN: llc -data-sections=1 -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s --check-prefix=DSECTIONS --check-prefix=CHECK
; RUN: llc -data-sections=0 -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s --check-prefix=NDSECTIONS --check-prefix=CHECK

@a = global i32 1
@b = global i32 2, !associated !0
!0 = !{i32* @a}
; CHECK-DAG: .section .data.b,"awo",@progbits,a

; Loop is OK. Also, normally -data-sections=0 would place @c and @d in the same section. !associated prevents that.
@c = global i32 2, !associated !2
@d = global i32 2, !associated !1
!1 = !{i32* @c}
!2 = !{i32* @d}
; CHECK-DAG: .section .data.c,"awo",@progbits,d
; CHECK-DAG: .section .data.d,"awo",@progbits,c

; BSS is OK.
@e = global i32 0
@f = global i32 0, !associated !3
@g = global i32 1, !associated !3
!3 = !{i32* @e}
; CHECK-DAG: .section .bss.f,"awo",@nobits,e
; CHECK-DAG: .section .data.g,"awo",@progbits,e

; Explicit sections.
@h = global i32 1, section "aaa"
@i = global i32 1, section "bbb", !associated !4
@j = global i32 1, section "bbb", !associated !4
@k = global i32 1, !associated !4
!4 = !{i32* @h}
; NDSECTIONS-DAG: .section	aaa,"aw",@progbits
; NDSECTIONS-DAG: .section	bbb,"awo",@progbits,h,unique,1
; NDSECTIONS-DAG: .section	bbb,"awo",@progbits,h,unique,2

; DSECTIONS-DAG: .section	aaa,"aw",@progbits,unique,1
; DSECTIONS-DAG: .section	bbb,"awo",@progbits,h,unique,2
; DSECTIONS-DAG: .section	bbb,"awo",@progbits,h,unique,3

; CHECK-DAG: .section	.data.k,"awo",@progbits,h

; Non-GlobalValue metadata.
@l = global i32 1, section "ccc", !associated !5
!5 = !{i32* null}
; CHECK-DAG: .section	ccc,"aw",@progbits

; Null metadata.
@m = global i32 1, section "ddd", !associated !6
!6 = distinct !{null}
; CHECK-DAG: .section	ddd,"aw",@progbits

; Aliases are OK.
@n = alias i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* @a to i64), i64 1297036692682702848) to i32*)
@o = global i32 1, section "eee", !associated !7
!7 = !{i32* @n}
; NDSECTIONS-DAG: .section	eee,"awo",@progbits,n,unique,3
; DSECTIONS-DAG: .section	eee,"awo",@progbits,n,unique,6
