; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

; CHECK: shift_parts_left_128
define void @shift_parts_left_128(i128* %val, i128* %amtptr) {
; CHECK: shl.b64
; CHECK: mov.u32
; CHECK: sub.s32
; CHECK: shr.u64
; CHECK: or.b64
; CHECK: add.s32
; CHECK: shl.b64
; CHECK: setp.gt.s32
; CHECK: selp.b64
; CHECK: shl.b64
  %amt = load i128, i128* %amtptr
  %a = load i128, i128* %val
  %val0 = shl i128 %a, %amt
  store i128 %val0, i128* %val
  ret void
}

; CHECK: shift_parts_right_128
define void @shift_parts_right_128(i128* %val, i128* %amtptr) {
; CHECK: shr.u64
; CHECK: sub.s32
; CHECK: shl.b64
; CHECK: or.b64
; CHECK: add.s32
; CHECK: shr.s64
; CHECK: setp.gt.s32
; CHECK: selp.b64
; CHECK: shr.s64
  %amt = load i128, i128* %amtptr
  %a = load i128, i128* %val
  %val0 = ashr i128 %a, %amt
  store i128 %val0, i128* %val
  ret void
}
