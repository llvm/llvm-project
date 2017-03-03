; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

@x = external global i32
@y = external global i32
@z = external global i32

define i32 @main() nounwind {
entry:
  store i32 1, i32* @x, align 4
  store i32 2148, i32* @y, align 4
  store i32 33332, i32* @z, align 4
  ret i32 0
}

; CHECK: li16   ${{[2-7]|16|17}}, 1
; CHECK: addiu  ${{[0-9]+}}, $zero, 2148
; CHECK: ori ${{[0-9]+}}, $zero, 33332
