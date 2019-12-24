; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands   < %s | FileCheck %s -check-prefix=load-relax

; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands -mips-constant-islands-no-load-relaxation  < %s | FileCheck %s -check-prefix=no-load-relax

; ModuleID = 'const6.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips--linux-gnu"

@i = common global i32 0, align 4
@j = common global i32 0, align 4
@k = common global i32 0, align 4
@l = common global i32 0, align 4
@b = common global i32 0, align 4

; Function Attrs: nounwind
define void @t() #0 {
entry:
  store i32 -559023410, i32* @i, align 4
; load-relax: 	lw	${{[0-9]+}}, $CPI0_0
; load-relax:	jrc	 $ra
; load-relax:	.p2align	2
; load-relax: $CPI0_0:
; load-relax:	.4byte	3735943886
; load-relax:	.end	t

; no-load-relax: lw	${{[0-9]+}}, $CPI0_1	# 16 bit inst
; no-load-relax:	jalrc 	${{[0-9]+}}
; no-load-relax:	b	$BB0_2
; no-load-relax:	.p2align	2
; no-load-relax: $CPI0_1:
; no-load-relax:	.4byte	3735943886
; no-load-relax: $BB0_2:

  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  call void bitcast (void (...)* @hoo to void ()*)()
  ret void
}

declare void @hoo(...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.4 (gitosis@dmz-portal.mips.com:clang.git b310439121c875937d78cc49cc969bc1197fc025) (gitosis@dmz-portal.mips.com:llvm.git 7fc0ca9656ebec8dad61f72f5a5ddfb232c070fd)"}


