; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@x = global i32 65504, align 4
@y = global i32 60929, align 4
@.str = private unnamed_addr constant [7 x i8] c"%08x \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32* @x, align 4
  %1 = load i32* @y, align 4
  %and = and i32 %0, %1
; 16:	and	${{[0-9]+}}, ${{[0-9]+}}
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([7 x i8]* @.str, i32 0, i32 0), i32 %and)
  ret i32 0
}

declare i32 @printf(i8*, ...)
