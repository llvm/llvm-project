; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands   < %s | FileCheck %s -check-prefix=b-short

; ModuleID = 'brsize3.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips--linux-gnu"

; Function Attrs: noreturn nounwind optsize
define void @foo() #0 {
entry:
  br label %x

x:                                                ; preds = %x, %entry
  tail call void asm sideeffect ".space 200", ""() #1, !srcloc !1
  br label %x
; b-short: $BB0_1:
; b-short:	#APP
; b-short:	.space 200
; b-short:	#NO_APP
; b-short:	b	$BB0_1 # 16 bit inst

}

attributes #0 = { noreturn nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { nounwind }

!1 = !{i32 45}
