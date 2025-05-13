; ModuleID = 'loadstore2.c'
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips--linux-gnu"

@c2 = common global i8 0, align 1
@c1 = common global i8 0, align 1
; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@s2 = common global i16 0, align 2
@s1 = common global i16 0, align 2
@i2 = common global i32 0, align 4
@i1 = common global i32 0, align 4
@f2 = common global float 0.000000e+00, align 4
@f1 = common global float 0.000000e+00, align 4
@d2 = common global double 0.000000e+00, align 8
@d1 = common global double 0.000000e+00, align 8

; Function Attrs: nounwind
define void @cfoo() #0 {
entry:
  %0 = load i8, ptr @c2, align 1
  store i8 %0, ptr @c1, align 1
; CHECK-LABEL:	cfoo:
; CHECK:	lbu	$[[REGc:[0-9]+]], 0(${{[0-9]+}})
; CHECK:	sb	$[[REGc]], 0(${{[0-9]+}})


  ret void
}

; Function Attrs: nounwind
define void @sfoo() #0 {
entry:
  %0 = load i16, ptr @s2, align 2
  store i16 %0, ptr @s1, align 2
; CHECK-LABEL:	sfoo:
; CHECK:	lhu	$[[REGs:[0-9]+]], 0(${{[0-9]+}})
; CHECK:	sh	$[[REGs]], 0(${{[0-9]+}})

  ret void
}

; Function Attrs: nounwind
define void @ifoo() #0 {
entry:
  %0 = load i32, ptr @i2, align 4
  store i32 %0, ptr @i1, align 4
; CHECK-LABEL:	ifoo:
; CHECK:	lw	$[[REGi:[0-9]+]], 0(${{[0-9]+}})
; CHECK:	sw	$[[REGi]], 0(${{[0-9]+}})

  ret void
}

; Function Attrs: nounwind
define void @ffoo() #0 {
entry:
  %0 = load float, ptr @f2, align 4
  store float %0, ptr @f1, align 4
; CHECK-LABEL:	ffoo:
; CHECK:	lwc1	$f[[REGf:[0-9]+]], 0(${{[0-9]+}})
; CHECK:	swc1	$f[[REGf]], 0(${{[0-9]+}})


  ret void
}

; Function Attrs: nounwind
define void @dfoo() #0 {
entry:
  %0 = load double, ptr @d2, align 8
  store double %0, ptr @d1, align 8
; CHECK-LABEL:        dfoo:
; CHECK:        ldc1    $f[[REGd:[0-9]+]], 0(${{[0-9]+}})
; CHECK:        sdc1    $f[[REGd]], 0(${{[0-9]+}})
; CHECK:        .end    dfoo
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }


