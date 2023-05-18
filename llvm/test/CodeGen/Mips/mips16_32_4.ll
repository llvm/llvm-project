; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=static -O3 < %s -mips-mixed-16-32  | FileCheck %s -check-prefix=16
; RUN: llc  -mtriple=mipsel -mcpu=mips32 -relocation-model=static -O3 < %s -mips-mixed-16-32  | FileCheck %s -check-prefix=32

define void @foo() #0 {
entry:
  ret void
}

; 16: 	.set	mips16
; 16: 	.ent	foo
; 16:	jrc $ra
; 16:	.end	foo
; 32: 	.set	mips16
; 32: 	.ent	foo
; 32:	jrc $ra
; 32:	.end	foo
define void @nofoo() #1 {
entry:
  ret void
}

; 16: 	.set	nomips16
; 16: 	.ent	nofoo
; 16:	.set	noreorder
; 16:	.set	nomacro
; 16:	.set	noat
; 16:	jr	$ra
; 16:	nop
; 16:	.set	at
; 16:	.set	macro
; 16:	.set	reorder
; 16:	.end	nofoo
; 32: 	.set	nomips16
; 32: 	.ent	nofoo
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	nop
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	nofoo
define i32 @main() #2 {
entry:
  ret i32 0
}

; 16: 	.set	mips16
; 16: 	.ent	main
; 16:	jrc $ra
; 16:	.end	main
; 32: 	.set	mips16
; 32: 	.ent	main
; 32:	jrc $ra
; 32:	.end	main


attributes #0 = { nounwind "less-precise-fpmad"="false" "mips16" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "nomips16" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "less-precise-fpmad"="false" "mips16" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
