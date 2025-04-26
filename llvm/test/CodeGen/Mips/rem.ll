; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@iiii = global i32 103, align 4
@jjjj = global i32 -4, align 4
@kkkk = common global i32 0, align 4


define void @test() nounwind {
entry:
  %0 = load i32, ptr @iiii, align 4
  %1 = load i32, ptr @jjjj, align 4
  %rem = srem i32 %0, %1
; 16:	div	$zero, ${{[0-9]+}}, ${{[0-9]+}}
; 16: 	mfhi	${{[0-9]+}}
  store i32 %rem, ptr @kkkk, align 4
  ret void
}


