; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s
; RUN: llc -march=mipsel -mcpu=mips32r6 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

@a = global i32 10, align 4
@b = global i32 0, align 4
@c = global i32 10, align 4
@d = global i32 0, align 4

define i32 @shift_left() nounwind {
entry:
  %0 = load i32, i32* @a, align 4
  %shl = shl i32 %0, 4
  store i32 %shl, i32* @b, align 4

  %1 = load i32, i32* @c, align 4
  %shl1 = shl i32 %1, 10
  store i32 %shl1, i32* @d, align 4

  ret i32 0
}

; CHECK: sll16  ${{[2-7]|16|17}}, ${{[2-7]|16|17}}, {{[0-7]}}
; CHECK: sll    ${{[0-9]+}}, ${{[0-9]+}}, {{[0-9]+}}

@i = global i32 10654, align 4
@j = global i32 0, align 4
@m = global i32 10, align 4
@n = global i32 0, align 4

define i32 @shift_right() nounwind {
entry:
  %0 = load i32, i32* @i, align 4
  %shr = lshr i32 %0, 4
  store i32 %shr, i32* @j, align 4

  %1 = load i32, i32* @m, align 4
  %shr1 = lshr i32 %1, 10
  store i32 %shr1, i32* @n, align 4

  ret i32 0
}

; CHECK: srl16  ${{[2-7]|16|17}}, ${{[2-7]|16|17}}, {{[0-7]}}
; CHECK: srl    ${{[0-9]+}}, ${{[0-9]+}}, {{[0-9]+}}
