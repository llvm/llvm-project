; RUN: llc  -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static  < %s | \
; RUN:       llvm-mc -arch=mipsel -mattr=+mips16 -show-inst | \
; RUN:       FileCheck %s 

@f = common global float 0.000000e+00, align 4

; Function Attrs: nounwind
define void @foo1() #0 {
entry:
  %c = alloca [10 x i8], align 1
  call void @x(ptr %c)
  call void @x(ptr %c)
  ret void
; CHECK: 	.ent	foo1
; CHECK: 	save	$16, $17, $ra, [[FS:[0-9]+]]  # <MCInst #[[#]] Save16
; CHECK: 	restore	$16, $17, $ra, [[FS]]   # <MCInst #[[#]] Restore16
; CHECK: 	.end	foo1
}

declare void @x(ptr) #1

; Function Attrs: nounwind
define void @foo2() #0 {
entry:
  %c = alloca [150 x i8], align 1
  call void @x(ptr %c)
  call void @x(ptr %c)
  ret void
; CHECK: 	.ent	foo2
; CHECK: 	save	$16, $17, $ra, [[FS:[0-9]+]]  # <MCInst #[[#]] SaveX16
; CHECK: 	restore	$16, $17, $ra, [[FS]]   # <MCInst #[[#]] RestoreX16
; CHECK: 	.end	foo2
}

; Function Attrs: nounwind
define void @foo3() #0 {
entry:
  %call = call float @xf()
  store float %call, ptr @f, align 4
  ret void
; CHECK: 	.ent	foo3
; CHECK: 	save	$16, $17, $18, $ra, [[FS:[0-9]+]]   # <MCInst #[[#]] SaveX16
; CHECK: 	restore	$16, $17, $18, $ra, [[FS]]    # <MCInst #[[#]] RestoreX16
; CHECK: 	.end	foo3
}

declare float @xf() #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }


