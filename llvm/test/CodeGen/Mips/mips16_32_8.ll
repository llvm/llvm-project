; RUN: llc  -mtriple=mipsel -mcpu=mips32 -relocation-model=static -O3 < %s -mips-mixed-16-32  | FileCheck %s -check-prefix=32

@x = global float 1.000000e+00, align 4
@y = global float 0x4007333340000000, align 4
@i = common global i32 0, align 4
@f = common global float 0.000000e+00, align 4
@.str = private unnamed_addr constant [8 x i8] c"f = %f\0A\00", align 1
@.str1 = private unnamed_addr constant [11 x i8] c"hello %i \0A\00", align 1
@.str2 = private unnamed_addr constant [13 x i8] c"goodbye %i \0A\00", align 1

define void @foo() #0 {
entry:
  store i32 10, ptr @i, align 4
  ret void
}

; 32: 	.set	mips16
; 32: 	.ent	foo
; 32:	jrc $ra
; 32:	.end	foo

define void @nofoo() #1 {
entry:
  store i32 20, ptr @i, align 4
  %0 = load float, ptr @x, align 4
  %1 = load float, ptr @y, align 4
  %add = fadd float %0, %1
  store float %add, ptr @f, align 4
  %2 = load float, ptr @f, align 4
  %conv = fpext float %2 to double
  %call = call i32 (ptr, ...) @printf(ptr @.str, double %conv)
  ret void
}

; 32: 	.set	nomips16
; 32: 	.ent	nofoo
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	add.s	{{.+}}
; 32:	mfc1    {{.+}}
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	nofoo
declare i32 @printf(ptr, ...) #2

define i32 @main() #3 {
entry:
  call void @foo()
  %0 = load i32, ptr @i, align 4
  %call = call i32 (ptr, ...) @printf(ptr @.str1, i32 %0)
  call void @nofoo()
  %1 = load i32, ptr @i, align 4
  %call1 = call i32 (ptr, ...) @printf(ptr @.str2, i32 %1)
  ret i32 0
}

; 32: 	.set	nomips16
; 32: 	.ent	main
; 32:	.set	noreorder
; 32:	.set	nomacro
; 32:	.set	noat
; 32:	jr	$ra
; 32:	.set	at
; 32:	.set	macro
; 32:	.set	reorder
; 32:	.end	main

attributes #0 = { nounwind "less-precise-fpmad"="false" "mips16" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "nomips16" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
