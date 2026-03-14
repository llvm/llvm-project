; RUN: llc -mcpu=cortex-a9 -mtriple=arm-linux-unknown -mattr=+neon < %s | FileCheck %s

; CHECK: func_4_8
; CHECK: vst1.32
; CHECK: bx lr
define void @func_4_8(<4 x i8> %param, ptr %p) {
  %r = add <4 x i8> %param, <i8 1, i8 2, i8 3, i8 4>
  store <4 x i8> %r, ptr %p
  ret void
}

; CHECK: func_2_16
; CHECK: vst1.32
; CHECK: bx lr
define void @func_2_16(<2 x i16> %param, ptr %p) {
  %r = add <2 x i16> %param, <i16 1, i16 2>
  store <2 x i16> %r, ptr %p
  ret void
}
