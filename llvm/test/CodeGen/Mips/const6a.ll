; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands   < %s | FileCheck %s -check-prefix=load-relax1

; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands   < %s | FileCheck %s -check-prefix=load-relax

; ModuleID = 'const6a.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips--linux-gnu"

@i = common global i32 0, align 4

; Function Attrs: nounwind
define void @t() #0 {
entry:
  store i32 -559023410, i32* @i, align 4
; load-relax-NOT: 	lw	${{[0-9]+}}, $CPI0_0 # 16 bit inst
; load-relax1: lw	${{[0-9]+}}, $CPI0_0
; load-relax:	jrc	 $ra
; load-relax:	.p2align	2
; load-relax: $CPI0_0:
; load-relax:	.4byte	3735943886
; load-relax:	.end	t
  call void asm sideeffect ".space 10000", ""() #1, !srcloc !1
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind }

!1 = !{i32 121}
