; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s -mips-mixed-16-32  | FileCheck %s 
; RUN: llc  -mtriple=mipsel -mcpu=mips32 -relocation-model=pic -O3 < %s -mips-mixed-16-32  | FileCheck %s 

define void @foo() #0 {
entry:
  ret void
}

; CHECK: 	.set	mips16
; CHECK:	.ent	foo
; CHECK:	jrc $ra
; CHECK:	.end	foo
attributes #0 = { nounwind "less-precise-fpmad"="false" "mips16" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
