; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 97, align 4
@c = common global i8 0, align 1
@.str = private unnamed_addr constant [8 x i8] c"%i %c \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32* @i, align 4
  %conv = trunc i32 %0 to i8
  store i8 %conv, i8* @c, align 1
  %1 = load i32* @i, align 4
  %2 = load i8* @c, align 1
  %conv1 = sext i8 %2 to i32
; 16:	sb	${{[0-9]+}}, 0(${{[0-9]+}})
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([8 x i8]* @.str, i32 0, i32 0), i32 %1, i32 %conv1)
  ret i32 0
}

declare i32 @printf(i8*, ...)
