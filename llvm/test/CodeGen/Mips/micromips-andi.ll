; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

@x = global i32 65504, align 4
@y = global i32 60929, align 4
@.str = private unnamed_addr constant [7 x i8] c"%08x \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, i32* @x, align 4
  %and1 = and i32 %0, 4
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds
                                  ([7 x i8], [7 x i8]* @.str, i32 0, i32 0), i32 %and1)

  %1 = load i32, i32* @y, align 4
  %and2 = and i32 %1, 5
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds
                                  ([7 x i8], [7 x i8]* @.str, i32 0, i32 0), i32 %and2)
  ret i32 0
}

declare i32 @printf(i8*, ...)

; CHECK: andi16 ${{[2-7]|16|17}}, ${{[2-7]|16|17}}
; CHECK: andi   ${{[0-9]+}}, ${{[0-9]+}}
