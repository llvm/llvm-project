; RUN: llc -mtriple=mipsel-w64-windows-gnu < %s | FileCheck %s -check-prefix=MIPSEL

define void @f() {
; MIPSEL-LABEL: f:
; MIPSEL:       # %bb.0: # %entry
; MIPSEL-NEXT:    addiu $sp, $sp, -24
; MIPSEL-NEXT:    sw $ra, 20($sp)
; MIPSEL-NEXT:    jal LeaveFoo
; MIPSEL-NEXT:    nop
; MIPSEL-NEXT:    jal LocalBar
; MIPSEL-NEXT:    nop
; MIPSEL-NEXT:    lw $ra, 20($sp)
; MIPSEL-NEXT:    jr $ra
; MIPSEL-NEXT:    addiu	$sp, $sp, 24

entry:
  call void @LeaveFoo()
  call void @LocalBar()
  ret void
}

declare void @LeaveFoo()
declare void @LocalBar()

