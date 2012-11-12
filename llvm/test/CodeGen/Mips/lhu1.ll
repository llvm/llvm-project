; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16


@s = global i16 255, align 2
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define i32 @main() nounwind {
entry:
  %i = alloca i32, align 4
  %0 = load i16* @s, align 2
  %conv = zext i16 %0 to i32
; 16:	lhu	${{[0-9]+}}, 0(${{[0-9]+}})
  store i32 %conv, i32* %i, align 4
  %1 = load i32* %i, align 4
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([5 x i8]* @.str, i32 0, i32 0), i32 %1)
  ret i32 0
}

declare i32 @printf(i8*, ...)
