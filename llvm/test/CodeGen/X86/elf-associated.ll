; RUN: llc -data-sections=1 -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s
; RUN: llc -data-sections=0 -mtriple x86_64-pc-linux-gnu < %s | FileCheck %s

@a = global i32 1
@b = global i32 2, !associated !0
!0 = !{ptr @a}
; CHECK-DAG: .section .data.b,"awo",@progbits,a

; Loop is OK. Also, normally -data-sections=0 would place @c and @d in the same section. !associated prevents that.
@c = global i32 2, !associated !2
@d = global i32 2, !associated !1
!1 = !{ptr @c}
!2 = !{ptr @d}
; CHECK-DAG: .section .data.c,"awo",@progbits,d
; CHECK-DAG: .section .data.d,"awo",@progbits,c

; BSS is OK.
@e = global i32 0
@f = global i32 0, !associated !3
@g = global i32 1, !associated !3
!3 = !{ptr @e}
; CHECK-DAG: .section .bss.f,"awo",@nobits,e
; CHECK-DAG: .section .data.g,"awo",@progbits,e

; Explicit sections.
@h = global i32 1, section "aaa"
@i = global i32 1, section "bbb", !associated !4
@j = global i32 1, section "bbb", !associated !4
@k = global i32 1, !associated !4
!4 = !{ptr @h}
; CHECK-DAG: .section	aaa,"aw",@progbits
; CHECK-DAG: .section	bbb,"awo",@progbits,h,unique,1
; CHECK-DAG: .section	bbb,"awo",@progbits,h,unique,2
; CHECK-DAG: .section	.data.k,"awo",@progbits,h

; Non-GlobalValue metadata.
@l = global i32 1, section "ccc", !associated !5
!5 = !{ptr null}
; CHECK-DAG: .section	ccc,"awo",@progbits,0,unique,3

; Null metadata.
@m = global i32 1, section "ddd", !associated !6
!6 = distinct !{null}
; CHECK-DAG: .section	ddd,"awo",@progbits,0,unique,4

; Aliases are OK.
@n = alias i32, ptr inttoptr (i64 add (i64 ptrtoint (ptr @a to i64), i64 1297036692682702848) to ptr)
@o = global i32 1, section "eee", !associated !7
!7 = !{ptr @n}
; CHECK-DAG: .section	eee,"awo",@progbits,n,unique,5
