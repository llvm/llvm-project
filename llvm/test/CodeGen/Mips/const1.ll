; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=static -mips16-constant-islands < %s | FileCheck %s

; ModuleID = 'const1.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mipsel-unknown-linux"

@i = common global i32 0, align 4
@j = common global i32 0, align 4
@k = common global i32 0, align 4
@l = common global i32 0, align 4

; Function Attrs: nounwind
define void @t() #0 {
entry:
  store i32 -559023410, ptr @i, align 4
  store i32 -559023410, ptr @j, align 4
  store i32 -87105875, ptr @k, align 4
  store i32 262991277, ptr @l, align 4
  ret void
; CHECK: 	lw	${{[0-9]+}}, $CPI0_0
; CHECK:	lw	${{[0-9]+}}, $CPI0_1
; CHECK: 	lw	${{[0-9]+}}, $CPI0_2
; CHECK: $CPI0_0:
; CHECK:	.4byte	3735943886
; CHECK: $CPI0_1:
; CHECK:	.4byte	4207861421
; CHECK: $CPI0_2:
; CHECK:	.4byte	262991277
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.4 (gitosis@dmz-portal.mips.com:clang.git b754974ec32ab712ea7d8b52cd8037b24e7d6ed3) (gitosis@dmz-portal.mips.com:llvm.git 8e211187b501bc73edb938fde0019c9a20bcffd5)"}
