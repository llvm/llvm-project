; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic  < %s | FileCheck %s -check-prefix=PIC

; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static  < %s | FileCheck %s -check-prefix=STATIC


@xi = common global i32 0, align 4
@x = common global float 0.000000e+00, align 4
@xd = common global double 0.000000e+00, align 8

; Function Attrs: nounwind
define void @it() #0 {
entry:
  %call = call i32 @i(i32 1)
  store i32 %call, i32* @xi, align 4
  ret void
; PIC: 	.ent	it
; STATIC: 	.ent	it
; PIC: 	save	$16, $17, $ra, [[FS:[0-9]+]]
; STATIC:      save	$16, $ra, [[FS:[0-9]+]]
; PIC: 	restore	$16, $17, $ra, [[FS]]
; STATIC:      restore	$16, $ra, [[FS]]
; PIC: 	.end	it
; STATIC: 	.end	it
}

declare i32 @i(i32) #1

; Function Attrs: nounwind
define void @ft() #0 {
entry:
  %call = call float @f()
  store float %call, float* @x, align 4
  ret void
; PIC: 	.ent	ft
; PIC: 	save	$16, $17, $ra, $18, [[FS:[0-9]+]]
; PIC: 	restore	$16, $17, $ra, $18, [[FS]]
; PIC: 	.end	ft
}

declare float @f() #1

; Function Attrs: nounwind
define void @dt() #0 {
entry:
  %call = call double @d()
  store double %call, double* @xd, align 8
  ret void
; PIC: 	.ent	dt
; PIC: 	save	$16, $17, $ra, $18, [[FS:[0-9]+]]
; PIC: 	restore	$16, $17, $ra, $18, [[FS]]
; PIC: 	.end	dt
}

declare double @d() #1

; Function Attrs: nounwind
define void @fft() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @ff(float %0)
  store float %call, float* @x, align 4
  ret void
; PIC: 	.ent	fft
; PIC: 	save	$16, $17, $ra, $18, [[FS:[0-9]+]]
; PIC: 	restore	$16, $17, $ra, $18, [[FS]]
; PIC: 	.end	fft
}

declare float @ff(float) #1

; Function Attrs: nounwind
define void @vft() #0 {
entry:
  %0 = load float, float* @x, align 4
  call void @vf(float %0)
  ret void
; PIC: 	.ent	vft
; STATIC: 	.ent	vft
; PIC: 	save	$16, $ra, [[FS:[0-9]+]]
; STATIC:      save	$16, $ra, [[FS:[0-9]+]]
; PIC: 	restore	$16, $ra, [[FS]]
; STATIC:      restore	$16, $ra, [[FS]]
; PIC: 	.end	vft
; STATIC: 	.end	vft
}

declare void @vf(float) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }


