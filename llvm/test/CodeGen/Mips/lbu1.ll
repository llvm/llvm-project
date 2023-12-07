; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@c = global i8 97, align 1
@.str = private unnamed_addr constant [5 x i8] c"%c \0A\00", align 1

define i32 @main() nounwind {
entry:
  %i = alloca i32, align 4
  %0 = load i8, ptr @c, align 1
  %conv = zext i8 %0 to i32
; 16:	lbu	${{[0-9]+}}, 0(${{[0-9]+}})
  store i32 %conv, ptr %i, align 4
  %1 = load i8, ptr @c, align 1
  %conv1 = zext i8 %1 to i32
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %conv1)
  ret i32 0
}

declare i32 @printf(ptr, ...)
