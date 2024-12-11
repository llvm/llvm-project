; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 97, align 4
@s = common global i16 0, align 2
@.str = private unnamed_addr constant [9 x i8] c"%i %hi \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, ptr @i, align 4
  %conv = trunc i32 %0 to i16
  store i16 %conv, ptr @s, align 2
  %1 = load i32, ptr @i, align 4
  %2 = load i16, ptr @s, align 2
  %conv1 = sext i16 %2 to i32
; 16:	sh	${{[0-9]+}}, 0(${{[0-9]+}})
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %1, i32 %conv1)
  ret i32 0
}

declare i32 @printf(ptr, ...)
