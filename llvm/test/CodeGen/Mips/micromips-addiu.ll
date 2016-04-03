; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

@x = global i32 65504, align 4
@y = global i32 60929, align 4
@z = global i32 60929, align 4
@.str = private unnamed_addr constant [7 x i8] c"%08x \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, i32* @x, align 4
  %addiu1 = add i32 %0, -7
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds
                                  ([7 x i8], [7 x i8]* @.str, i32 0, i32 0), i32 %addiu1)

  %1 = load i32, i32* @y, align 4
  %addiu2 = add i32 %1, 55
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds
                                  ([7 x i8], [7 x i8]* @.str, i32 0, i32 0), i32 %addiu2)

  %2 = load i32, i32* @z, align 4
  %addiu3 = add i32 %2, 24
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds
                                  ([7 x i8], [7 x i8]* @.str, i32 0, i32 0), i32 %addiu3)
  ret i32 0
}

declare i32 @printf(i8*, ...)

; CHECK: addius5  ${{[0-9]+}}, -7
; CHECK: addiu    ${{[0-9]+}}, ${{[0-9]+}}, 55
; CHECK: addiur2  ${{[2-7]|16|17}}, ${{[2-7]|16|17}}, 24
