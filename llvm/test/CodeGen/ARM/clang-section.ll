;RUN: llc -mtriple=armv7-eabi %s -o - | FileCheck %s
;Test that global variables and functions are assigned to correct sections.

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-arm-none-eabi"

@a = global i32 0, align 4 #0
@b = global i32 1, align 4 #0
@c = global [4 x i32] zeroinitializer, align 4 #0
@d = global [5 x i16] zeroinitializer, align 2 #0
@e = global [6 x i16] [i16 0, i16 0, i16 1, i16 0, i16 0, i16 0], align 2 #0
@f = constant i32 2, align 4 #0
@h = global i32 0, align 4 #1
@i = global i32 0, align 4 #2
@j = constant i32 4, align 4 #2
@k = global i32 0, align 4 #2
@_ZZ3gooE7lstat_h = internal global i32 0, align 4 #2
@_ZL1g = internal global [2 x i32] zeroinitializer, align 4 #0
@l = global i32 5, align 4 #3
@m = constant i32 6, align 4 #3
@n = global i32 0, align 4
@o = global i32 6, align 4
@p = constant i32 7, align 4

declare i32 @zoo(ptr, ptr) #6

; Function Attrs: noinline nounwind
define i32 @hoo() #7 {
entry:
  %0 = load i32, ptr @b, align 4
  ret i32 %0
}

attributes #0 = { "bss-section"="my_bss.1" "data-section"="my_data.1" "rodata-section"="my_rodata.1" }
attributes #1 = { "data-section"="my_data.1" "rodata-section"="my_rodata.1" }
attributes #2 = { "bss-section"="my_bss.2" "rodata-section"="my_rodata.1" }
attributes #3 = { "bss-section"="my_bss.2" "data-section"="my_data.2" "rodata-section"="my_rodata.2" }
attributes #6 = { "correctly-rounded-divide-sqrt-fp-math"="false" "denormal-fp-math"="preserve-sign,preserve-sign" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a9" "target-features"="+dsp,+fp16,+neon,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denormal-fp-math"="preserve-sign,preserve-sign" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a9" "target-features"="+dsp,+fp16,+neon,+vfp3" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"static_rwdata", i32 1}
!2 = !{i32 1, !"enumsize_buildattr", i32 2}
!3 = !{i32 1, !"armlib_unavailable", i32 0}

;CHECK: 	.text
;CHECK: 	.type	hoo,%function
;CHECK: hoo:

;CHECK: 	.type	a,%object
;CHECK: 	.section	my_bss.1,"aw",%nobits
;CHECK: a:

;CHECK: 	.type	b,%object
;CHECK: 	.section	my_data.1,"aw",%progbits
;CHECK: b:

;CHECK: 	.type	c,%object
;CHECK: 	.section	my_bss.1,"aw",%nobits
;CHECK: c:

;CHECK: 	.type	d,%object
;CHECK: d:

;CHECK: 	.type	e,%object
;CHECK: 	.section	my_data.1,"aw",%progbits
;CHECK: e:

;CHECK: 	.type	f,%object
;CHECK: 	.section	my_rodata.1,"a",%progbits
;CHECK: f:

;CHECK: 	.type	h,%object
;CHECK: 	.bss
;CHECK: h:

;CHECK: 	.type	i,%object
;CHECK: 	.section	my_bss.2,"aw",%nobits
;CHECK: i:

;CHECK: 	.type	j,%object
;CHECK: 	.section	my_rodata.1,"a",%progbits
;CHECK: j:

;CHECK: 	.type	k,%object
;CHECK: 	.section	my_bss.2,"aw",%nobits
;CHECK: k:

;CHECK: 	.type	_ZZ3gooE7lstat_h,%object @ @_ZZ3gooE7lstat_h
;CHECK: _ZZ3gooE7lstat_h:

;CHECK: 	.type	_ZL1g,%object
;CHECK: 	.section	my_bss.1,"aw",%nobits
;CHECK: _ZL1g:

;CHECK: 	.type	l,%object
;CHECK: 	.section	my_data.2,"aw",%progbits
;CHECK: l:

;CHECK: 	.type	m,%object
;CHECK: 	.section	my_rodata.2,"a",%progbits
;CHECK: m:

;CHECK: 	.type	n,%object
;CHECK: 	.bss
;CHECK: n:

;CHECK: 	.type	o,%object
;CHECK: 	.data
;CHECK: o:

;CHECK: 	.type	p,%object
;CHECK: 	.section	.rodata,"a",%progbits
;CHECK: p:
