; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32 -relocation-model=static -O3 < %s -mips-os16  | FileCheck %s -check-prefix=32

; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mcpu=mips32 -relocation-model=static -O3 -mips16-constant-islands < %s -mips-os16  | FileCheck %s -check-prefix=cisle

@i = global i32 1, align 4
@f = global float 1.000000e+00, align 4

define void @vv() #0 {
entry:
  ret void
}

; 32: 	.set	mips16
; 32: 	.ent	vv

; 32:	save	{{.+}}
; 32:	restore	{{.+}} 
; 32:	.end	vv

define i32 @iv() #0 {
entry:
  %0 = load i32, i32* @i, align 4
  ret i32 %0
}

; 32: 	.set	mips16
; 32: 	.ent	iv

; 32:	save	{{.+}}
; 32:	restore	{{.+}} 
; 32:	.end	iv

define void @vif(i32 %i, float %f) #0 {
entry:
  %i.addr = alloca i32, align 4
  %f.addr = alloca float, align 4
  store i32 %i, i32* %i.addr, align 4
  store float %f, float* %f.addr, align 4
  ret void
}

; 32: 	.set	mips16
; 32: 	.ent	vif

; 32:	save	{{.+}}
; 32:	restore	{{.+}} 
; 32:	.end	vif

define void @foo() #0 {
entry:
  store float 2.000000e+00, float* @f, align 4
  ret void
}

; 32: 	.set	mips16
; 32: 	.ent	foo

; 32:	save	{{.+}}
; 32:	restore	{{.+}} 
; 32:	.end	foo

; cisle:	.end	foo

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }


define float @fv() #0 {
entry:
  ret float 1.000000e+00
}

; 32: 	.set	nomips16
; 32: 	.ent	fv
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	fv
