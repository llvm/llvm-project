; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -relocation-model=static -mips32-function-mask=10 -mips-os16 < %s | FileCheck %s -check-prefix=fmask1

; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -relocation-model=static -mips32-function-mask=01 -mips-os16 < %s | FileCheck %s -check-prefix=fmask2

; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -relocation-model=static -mips32-function-mask=10. -mips-os16 < %s | FileCheck %s -check-prefix=fmask1nr

; Function Attrs: nounwind optsize readnone
define void @foo1()  {
entry:
  ret void
; fmask1: .ent foo1
; fmask1: .set	noreorder
; fmask1: .set	nomacro
; fmask1: .set	noat
; fmask1: .set	at
; fmask1: .set	macro
; fmask1: .set	reorder
; fmask1: .end	foo1
; fmask2: .ent	foo1
; fmask2: jrc $ra
; fmask2: .end	foo1
; fmask1nr: .ent foo1
; fmask1nr: .set	noreorder
; fmask1nr: .set	nomacro
; fmask1nr: .set	noat
; fmask1nr: .set	at
; fmask1nr: .set	macro
; fmask1nr: .set	reorder
; fmask1nr: .end	foo1
}

; Function Attrs: nounwind optsize readnone
define void @foo2()  {
entry:
  ret void
; fmask2: .ent foo2
; fmask2: .set	noreorder
; fmask2: .set	nomacro
; fmask2: .set	noat
; fmask2: .set	at
; fmask2: .set	macro
; fmask2: .set	reorder
; fmask2: .end	foo2
; fmask1: .ent	foo2
; fmask1: jrc $ra
; fmask1: .end	foo2
; fmask1nr: .ent	foo2
; fmask1nr: jrc $ra
; fmask1nr: .end	foo2
}

; Function Attrs: nounwind optsize readnone
define void @foo3()  {
entry:
  ret void
; fmask1: .ent foo3
; fmask1: .set	noreorder
; fmask1: .set	nomacro
; fmask1: .set	noat
; fmask1: .set	at
; fmask1: .set	macro
; fmask1: .set	reorder
; fmask1: .end	foo3
; fmask2:  .ent	foo3
; fmask2:  jrc $ra
; fmask2:  .end	foo3
; fmask1r:  .ent	foo3
; fmask1r:  jrc $ra
; fmask1r:  .end	foo3
}

; Function Attrs: nounwind optsize readnone
define void @foo4()  {
entry:
  ret void
; fmask2: .ent foo4
; fmask2: .set	noreorder
; fmask2: .set	nomacro
; fmask2: .set	noat
; fmask2: .set	at
; fmask2: .set	macro
; fmask2: .set	reorder
; fmask2: .end	foo4
; fmask1: .ent	foo4
; fmask1: jrc $ra
; fmask1: .end	foo4
; fmask1nr: .ent	foo4
; fmask1nr: jrc $ra
; fmask1nr: .end	foo4
}


