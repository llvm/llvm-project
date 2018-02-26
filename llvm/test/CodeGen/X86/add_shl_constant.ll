; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s

; CHECK-LABEL: add_shl_add_constant_1_i32
; CHECK: leal 984(%rsi,%rdi,8), %eax
; CHECK-NEXT: retq
define i32 @add_shl_add_constant_1_i32(i32 %x, i32 %y) nounwind {
  %add.0 = add i32 %x, 123
  %shl = shl i32 %add.0, 3
  %add.1 = add i32 %shl, %y
  ret i32 %add.1
}

; CHECK-LABEL: add_shl_add_constant_2_i32
; CHECK: leal 984(%rsi,%rdi,8), %eax
; CHECK-NEXT: retq
define i32 @add_shl_add_constant_2_i32(i32 %x, i32 %y) nounwind {
  %add.0 = add i32 %x, 123
  %shl = shl i32 %add.0, 3
  %add.1 = add i32 %y, %shl
  ret i32 %add.1
}

; CHECK: LCPI2_0:
; CHECK: .long 984
; CHECK: _add_shl_add_constant_1_v4i32
; CHECK: pslld $3, %[[REG:xmm[0-9]+]]
; CHECK: paddd %xmm1, %[[REG]]
; CHECK: paddd LCPI2_0(%rip), %[[REG:xmm[0-9]+]]
; CHECK: retq
define <4 x i32> @add_shl_add_constant_1_v4i32(<4 x i32> %x, <4 x i32> %y) nounwind {
  %add.0 = add <4 x i32> %x, <i32 123, i32 123, i32 123, i32 123>
  %shl = shl <4 x i32> %add.0, <i32 3, i32 3, i32 3, i32 3>
  %add.1 = add <4 x i32> %shl, %y
  ret <4 x i32> %add.1
}

; CHECK: LCPI3_0:
; CHECK: .long 984
; CHECK: _add_shl_add_constant_2_v4i32
; CHECK: pslld $3, %[[REG:xmm[0-9]+]]
; CHECK: paddd %xmm1, %[[REG]]
; CHECK: paddd LCPI3_0(%rip), %[[REG:xmm[0-9]+]]
; CHECK: retq
define <4 x i32> @add_shl_add_constant_2_v4i32(<4 x i32> %x, <4 x i32> %y) nounwind {
  %add.0 = add <4 x i32> %x, <i32 123, i32 123, i32 123, i32 123>
  %shl = shl <4 x i32> %add.0, <i32 3, i32 3, i32 3, i32 3>
  %add.1 = add <4 x i32> %y, %shl
  ret <4 x i32> %add.1
}
