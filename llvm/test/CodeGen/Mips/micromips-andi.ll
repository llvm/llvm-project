; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

@x = global i32 65504, align 4
@y = global i32 60929, align 4
@.str = private unnamed_addr constant [7 x i8] c"%08x \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, ptr @x, align 4
  %and1 = and i32 %0, 4
  %call1 = call i32 (ptr, ...) @printf(ptr @.str, i32 %and1)

  %1 = load i32, ptr @y, align 4
  %and2 = and i32 %1, 5
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, i32 %and2)
  ret i32 0
}

declare i32 @printf(ptr, ...)

; CHECK: andi16 ${{[2-7]|16|17}}, ${{[2-7]|16|17}}
; CHECK: andi   ${{[0-9]+}}, ${{[0-9]+}}
