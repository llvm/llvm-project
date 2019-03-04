; RUN: llc -o - %s -mtriple=aarch64-windows -aarch64-enable-compress-jump-tables=0 | FileCheck %s

define void @f(i32 %x) {
entry:
  switch i32 %x, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:                                            ; preds = %entry
  tail call void @g(i32 0) #2
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  tail call void @g(i32 1) #2
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  tail call void @g(i32 2) #2
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  tail call void @g(i32 3) #2
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb
  tail call void @g(i32 10) #2
  ret void
}

declare void @g(i32)

; CHECK:		.text
; CHECK:		f:
; CHECK:		.seh_proc f
; CHECK:		b	g
; CHECK-NEXT:	.p2align	2
; CHECK-NEXT:	.LJTI0_0:
; CHECK:		.word	.LBB0_2-.LJTI0_0
; CHECK:		.word	.LBB0_3-.LJTI0_0
; CHECK:		.word	.LBB0_4-.LJTI0_0
; CHECK:		.word	.LBB0_5-.LJTI0_0
; CHECK:		.section	.xdata,"dr"
; CHECK:		.seh_handlerdata
; CHECK:		.text
; CHECK:		.seh_endproc
