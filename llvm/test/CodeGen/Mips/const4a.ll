; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands -mips-constant-islands-no-load-relaxation < %s | FileCheck %s -check-prefix=no-load-relax

; ModuleID = 'const4.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips--linux-gnu"

@i = common global i32 0, align 4
@b = common global i32 0, align 4
@j = common global i32 0, align 4
@k = common global i32 0, align 4
@l = common global i32 0, align 4

; Function Attrs: nounwind
define void @t() #0 {
entry:
  store i32 -559023410, ptr @i, align 4
  %0 = load i32, ptr @b, align 4
; no-load-relax:	lw	${{[0-9]+}}, $CPI0_1	# 16 bit inst
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else
; no-load-relax:	beqz	${{[0-9]+}}, $BB0_3
; no-load-relax:	lw	${{[0-9]+}}, %call16(foo)(${{[0-9]+}})
; no-load-relax:	b	$BB0_4
; no-load-relax:	.p2align	2
; no-load-relax: $CPI0_1:
; no-load-relax:	.4byte	3735943886
; no-load-relax: $BB0_3:
; no-load-relax:	lw	${{[0-9]+}}, %call16(goo)(${{[0-9]+}})
if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.else:                                          ; preds = %entry
  call void @goo()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  call void @hoo()
  ret void
}

declare void @foo(...) #1

declare void @goo(...) #1

declare void @hoo(...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.4 (gitosis@dmz-portal.mips.com:clang.git b310439121c875937d78cc49cc969bc1197fc025) (gitosis@dmz-portal.mips.com:llvm.git 7fc0ca9656ebec8dad61f72f5a5ddfb232c070fd)"}
