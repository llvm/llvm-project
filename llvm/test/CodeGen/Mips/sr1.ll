; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -relocation-model=static < %s | FileCheck %s

; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -relocation-model=static < %s | FileCheck %s -check-prefix=NEG

@f = common global float 0.000000e+00, align 4

; Function Attrs: nounwind
define void @foo1() #0 {
entry:
  %c = alloca [10 x i8], align 1
  call void @x(ptr %c)
  call void @x(ptr %c)
  ret void
; CHECK: 	.ent	foo1
; CHECK: 	save	$16, $17, $ra, [[FS:[0-9]+]]  # 16 bit inst
; CHECK: 	restore	$16, $17, $ra, [[FS]] # 16 bit inst
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
; CHECK: 	save	$16, $17, $ra, [[FS:[0-9]+]] 
; CHECK: 	restore	$16, $17, $ra, [[FS]] 
; CHECK: 	.end	foo2
}

; Function Attrs: nounwind
define void @foo3() #0 {
entry:
  %call = call float @xf()
  store float %call, ptr @f, align 4
  ret void
; CHECK: 	.ent	foo3
; CHECK: 	save	$16, $17, $ra, $18, [[FS:[0-9]+]]
; CHECK: 	restore	$16, $17, $ra, $18, [[FS]]
; CHECK: 	.end	foo3
; NEG: 	.ent	foo3
; NEG-NOT: 	save	$16, $17, $ra, $18, {{[0-9]+}} # 16 bit inst
; NEG-NOT: 	restore	$16, $17, $ra, $18, {{[0-9]+}} # 16 bit inst
; NEG: 	.end	foo3
}

declare float @xf() #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }


