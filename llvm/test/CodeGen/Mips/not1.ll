; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@x = global i32 65504, align 4
@y = global i32 60929, align 4
@.str = private unnamed_addr constant [7 x i8] c"%08x \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, ptr @x, align 4
  %neg = xor i32 %0, -1
; 16:	not	${{[0-9]+}}, ${{[0-9]+}}
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %neg)
  ret i32 0
}

declare i32 @printf(ptr, ...)
