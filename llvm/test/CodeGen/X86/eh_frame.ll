; RUN: llc < %s -mtriple x86_64-unknown-linux-gnu | FileCheck -check-prefix=STATIC %s
; RUN: llc < %s -mtriple x86_64-unknown-linux-gnu -relocation-model=pic | FileCheck -check-prefix=PIC %s

@__FRAME_END__ = constant [1 x i32] zeroinitializer, section ".eh_frame"

@foo = external global i32
@bar1 = constant ptr @foo, section "my_bar1", align 8


; STATIC: .section	.eh_frame,"a",@unwind
; STATIC: .section	my_bar1,"a",@progbits

; PIC:	.section	.eh_frame,"a",@unwind
; PIC:	.section	my_bar1,"aw",@progbits
