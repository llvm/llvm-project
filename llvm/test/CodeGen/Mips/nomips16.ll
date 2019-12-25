; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static < %s | FileCheck %s

@x = global float 0.000000e+00, align 4
@.str = private unnamed_addr constant [20 x i8] c"in main: mips16 %f\0A\00", align 1

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  %0 = load float, float* @x, align 4
  %conv = fpext float %0 to double
  %add = fadd double %conv, 1.500000e+00
  %conv1 = fptrunc double %add to float
  store float %conv1, float* @x, align 4
  ret void
}
; CHECK: 	.ent	foo
; CHECK: 	jal	__mips16_extendsfdf2
; CHECK:   	.end	foo

; Function Attrs: nounwind
define void @nofoo() #1 {
entry:
  %0 = load float, float* @x, align 4
  %conv = fpext float %0 to double
  %add = fadd double %conv, 3.900000e+00
  %conv1 = fptrunc double %add to float
  store float %conv1, float* @x, align 4
  ret void
}

; CHECK: 	.ent	nofoo
; CHECK: 	cvt.d.s	$f{{.+}}, $f{{.+}}
; CHECK: 	.end	nofoo


attributes #0 = { nounwind "less-precise-fpmad"="false" "mips16" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "nomips16" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

