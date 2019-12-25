; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck %s -check-prefix=pic
; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -relocation-model=static -mips32-function-mask=1010111 -mips-os16 < %s | FileCheck %s -check-prefix=fmask

@x = global float 1.500000e+00, align 4
@xn = global float -1.900000e+01, align 4
@negone = global float -1.000000e+00, align 4
@one = global float 1.000000e+00, align 4
@xd = global double 0x40048B0A8EA4481E, align 8
@xdn = global double 0xC0311F9ADD373963, align 8
@negoned = global double -1.000000e+00, align 8
@oned = global float 1.000000e+00, align 4
@y = common global float 0.000000e+00, align 4
@yd = common global double 0.000000e+00, align 8

; Function Attrs: nounwind
define void @foo1() #0 {
; fmask: .ent foo1
; fmask: .set	noreorder
; fmask: .set	nomacro
; fmask: .set	noat
; fmask: .set	at
; fmask: .set	macro
; fmask: .set	reorder
; fmask: .end	foo1
entry:
  %0 = load float, float* @x, align 4
  %1 = load float, float* @one, align 4
  %call = call float @copysignf(float %0, float %1) #2
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @copysignf(float, float) #1

; Function Attrs: nounwind
define void @foo2() #0 {
; fmask:	.ent	foo2
; fmask:	save	{{.*}}
; fmask:	.end	foo2
entry:
  %0 = load float, float* @x, align 4
  %1 = load float, float* @negone, align 4
  %call = call float @copysignf(float %0, float %1) #2
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind
define void @foo3() #0 {
entry:
; fmask: .ent foo3
; fmask: .set	noreorder
; fmask: .set	nomacro
; fmask: .set	noat
; fmask: .set	at
; fmask: .set	macro
; fmask: .set	reorder
; fmask: .end	foo3
  %0 = load double, double* @xd, align 8
  %1 = load float, float* @oned, align 4
  %conv = fpext float %1 to double
  %call = call double @copysign(double %0, double %conv) #2
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @copysign(double, double) #1

; Function Attrs: nounwind
define void @foo4() #0 {
entry:
; fmask:	.ent	foo4
; fmask:	save	{{.*}}
; fmask:	.end	foo4
  %0 = load double, double* @xd, align 8
  %1 = load double, double* @negoned, align 8
  %call = call double @copysign(double %0, double %1) #2
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind
define void @foo5() #0 {
entry:
  %0 = load float, float* @xn, align 4
  %call = call float @fabsf(float %0) #2
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @fabsf(float) #1

; Function Attrs: nounwind
define void @foo6() #0 {
entry:
  %0 = load double, double* @xdn, align 8
  %call = call double @fabs(double %0) #2
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @fabs(double) #1

; Function Attrs: nounwind
define void @foo7() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @sinf(float %0) #3
;pic:	lw	${{[0-9]+}}, %call16(sinf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind
declare float @sinf(float) #0

; Function Attrs: nounwind
define void @foo8() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @sin(double %0) #3
;pic:	lw	${{[0-9]+}}, %call16(sin)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind
declare double @sin(double) #0

; Function Attrs: nounwind
define void @foo9() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @cosf(float %0) #3
;pic:	lw	${{[0-9]+}}, %call16(cosf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind
declare float @cosf(float) #0

; Function Attrs: nounwind
define void @foo10() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @cos(double %0) #3
;pic:	lw	${{[0-9]+}}, %call16(cos)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind
declare double @cos(double) #0

; Function Attrs: nounwind
define void @foo11() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @sqrtf(float %0) #3
;pic:	lw	${{[0-9]+}}, %call16(sqrtf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind
declare float @sqrtf(float) #0

; Function Attrs: nounwind
define void @foo12() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @sqrt(double %0) #3
;pic:	lw	${{[0-9]+}}, %call16(sqrt)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind
declare double @sqrt(double) #0

; Function Attrs: nounwind
define void @foo13() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @floorf(float %0) #2
;pic:	lw	${{[0-9]+}}, %call16(floorf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @floorf(float) #1

; Function Attrs: nounwind
define void @foo14() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @floor(double %0) #2
;pic:	lw	${{[0-9]+}}, %call16(floor)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @floor(double) #1

; Function Attrs: nounwind
define void @foo15() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @nearbyintf(float %0) #2
;pic:	lw	${{[0-9]+}}, %call16(nearbyintf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @nearbyintf(float) #1

; Function Attrs: nounwind
define void @foo16() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @nearbyint(double %0) #2
;pic:	lw	${{[0-9]+}}, %call16(nearbyint)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @nearbyint(double) #1

; Function Attrs: nounwind
define void @foo17() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @ceilf(float %0) #2
;pic:	lw	${{[0-9]+}}, %call16(ceilf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @ceilf(float) #1

; Function Attrs: nounwind
define void @foo18() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @ceil(double %0) #2
;pic:	lw	${{[0-9]+}}, %call16(ceil)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @ceil(double) #1

; Function Attrs: nounwind
define void @foo19() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @rintf(float %0) #2
;pic:	lw	${{[0-9]+}}, %call16(rintf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @rintf(float) #1

; Function Attrs: nounwind
define void @foo20() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @rint(double %0) #2
;pic:	lw	${{[0-9]+}}, %call16(rint)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @rint(double) #1

; Function Attrs: nounwind
define void @foo21() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @truncf(float %0) #2
;pic:	lw	${{[0-9]+}}, %call16(truncf)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @truncf(float) #1

; Function Attrs: nounwind
define void @foo22() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @trunc(double %0) #2
;pic:	lw	${{[0-9]+}}, %call16(trunc)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare double @trunc(double) #1

; Function Attrs: nounwind
define void @foo23() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @log2f(float %0) #3
;pic:	lw	${{[0-9]+}}, %call16(log2f)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind
declare float @log2f(float) #0

; Function Attrs: nounwind
define void @foo24() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @log2(double %0) #3
;pic:	lw	${{[0-9]+}}, %call16(log2)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind
declare double @log2(double) #0

; Function Attrs: nounwind
define void @foo25() #0 {
entry:
  %0 = load float, float* @x, align 4
  %call = call float @exp2f(float %0) #3
;pic:	lw	${{[0-9]+}}, %call16(exp2f)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_sf_1)(${{[0-9]+}})
  store float %call, float* @y, align 4
  ret void
}

; Function Attrs: nounwind
declare float @exp2f(float) #0

; Function Attrs: nounwind
define void @foo26() #0 {
entry:
  %0 = load double, double* @xd, align 8
  %call = call double @exp2(double %0) #3
;pic:	lw	${{[0-9]+}}, %call16(exp2)(${{[0-9]+}})
;pic:	lw	${{[0-9]+}}, %got(__mips16_call_stub_df_2)(${{[0-9]+}})
  store double %call, double* @yd, align 8
  ret void
}

; Function Attrs: nounwind
declare double @exp2(double) #0

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }
