; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=static -O3 < %s -mips-mixed-16-32  | FileCheck %s -check-prefix=16

define void @foo() #0 {
entry:
  ret void
}
; 16: 	.set	nomips16
; 16: 	.ent	foo
; 16:	.set	noreorder
; 16:	.set	nomacro
; 16:	.set	noat
; 16:	jr	$ra
; 16:	nop
; 16:	.set	at
; 16:	.set	macro
; 16:	.set	reorder
; 16:	.end	foo

define void @nofoo() #1 {
entry:
  ret void
}

; 16: 	.set	mips16
; 16: 	.ent	nofoo

; 16:	jrc $ra
; 16:	.end	nofoo

define i32 @main() #2 {
entry:
  ret i32 0
}

; 16: 	.set	nomips16
; 16: 	.ent	main
; 16:	.set	noreorder
; 16:	.set	nomacro
; 16:	.set	noat
; 16:	jr	$ra
; 16:	.set	at
; 16:	.set	macro
; 16:	.set	reorder
; 16:	.end	main











attributes #0 = { nounwind "less-precise-fpmad"="false" "nomips16" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false"  "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "less-precise-fpmad"="false" "nomips16" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
