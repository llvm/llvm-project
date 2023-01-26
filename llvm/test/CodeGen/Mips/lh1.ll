; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@s = global i16 -1, align 2
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define i32 @main() nounwind {
entry:
  %i = alloca i32, align 4
  %0 = load i16, ptr @s, align 2
  %conv = sext i16 %0 to i32
; 16:	lh	${{[0-9]+}}, 0(${{[0-9]+}})
  store i32 %conv, ptr %i, align 4
  %1 = load i32, ptr %i, align 4
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %1)
  ret i32 0
}

declare i32 @printf(ptr, ...)
