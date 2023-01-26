; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 -354, align 4
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, ptr @i, align 4
  %shr = ashr i32 %0, 3
; 16:	sra	${{[0-9]+}}, ${{[0-9]+}}, {{[0-9]+}}
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %shr)
  ret i32 0
}

declare i32 @printf(ptr, ...)
