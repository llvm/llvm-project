; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 10, align 4
@j = global i32 20, align 4
@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, ptr @j, align 4
  %1 = load i32, ptr @i, align 4
  %sub = sub nsw i32 %0, %1
; 16:	subu	${{[0-9]+}}, ${{[0-9]+}}, ${{[0-9]+}}
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %sub)
  ret i32 0
}

declare i32 @printf(ptr, ...)
