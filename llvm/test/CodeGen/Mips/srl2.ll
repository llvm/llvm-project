; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 10654, align 4
@j = global i32 0, align 4
@k = global i32 4, align 4
@.str = private unnamed_addr constant [5 x i8] c"%i \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32* @i, align 4
  %1 = load i32* @k, align 4
  %shr = lshr i32 %0, %1
; 16:	srlv	${{[0-9]+}}, ${{[0-9]+}}
  store i32 %shr, i32* @j, align 4
  %2 = load i32* @j, align 4
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([5 x i8]* @.str, i32 0, i32 0), i32 %2)
  ret i32 0
}

declare i32 @printf(i8*, ...)
