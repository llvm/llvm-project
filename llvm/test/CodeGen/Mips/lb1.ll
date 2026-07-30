; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@c = global i8 -1, align 1
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define i32 @main() nounwind {
entry:
  %i = alloca i32, align 4
  %0 = load i8, ptr @c, align 1
; 16:	lb	${{[0-9]+}}, 0(${{[0-9]+}})
  %conv = sext i8 %0 to i32
  store i32 %conv, ptr %i, align 4
  %1 = load i32, ptr %i, align 4
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %1)
  ret i32 0
}

declare i32 @printf(ptr, ...)
