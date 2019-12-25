; RUN: llc  -march=mipsel -mcpu=mips32 -relocation-model=static -O3 < %s -mips-os16  | FileCheck %s -check-prefix=32

@x = global float 1.000000e+00, align 4
@y = global float 2.000000e+00, align 4
@zz = common global float 0.000000e+00, align 4
@z = common global float 0.000000e+00, align 4

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

define double @dv() #0 {
entry:
  ret double 2.000000e+00
}

; 32: 	.set	nomips16
; 32: 	.ent	dv
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	dv

define void @vf(float %x) #0 {
entry:
  %x.addr = alloca float, align 4
  store float %x, float* %x.addr, align 4
  ret void
}

; 32: 	.set	nomips16
; 32: 	.ent	vf
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	vf

define void @vd(double %x) #0 {
entry:
  %x.addr = alloca double, align 8
  store double %x, double* %x.addr, align 8
  ret void
}

; 32: 	.set	nomips16
; 32: 	.ent	vd
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	vd

define void @foo1() #0 {
entry:
  store float 1.000000e+00, float* @zz, align 4
  %0 = load float, float* @y, align 4
  %1 = load float, float* @x, align 4
  %add = fadd float %0, %1
  store float %add, float* @z, align 4
  ret void
}

; 32: 	.set	nomips16
; 32: 	.ent	foo1
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	foo1

define void @foo2() #0 {
entry:
  %0 = load float, float* @x, align 4
  call void @vf(float %0)
  ret void
}


; 32: 	.set	nomips16
; 32: 	.ent	foo2
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	foo2

define void @foo3() #0 {
entry:
  %call = call float @fv()
  store float %call, float* @x, align 4
  ret void
}

; 32: 	.set	nomips16
; 32: 	.ent	foo3
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	foo3

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

define void @vv() #0 {
entry:
  ret void
}

; 32: 	.set	mips16
; 32: 	.ent	vv

; 32:	save	{{.+}}
; 32:	restore	{{.+}} 
; 32:	.end	vv



