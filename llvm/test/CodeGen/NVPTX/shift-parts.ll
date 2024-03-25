; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK: shift_parts_left_128
define void @shift_parts_left_128(ptr %val, ptr %amtptr) {
; CHECK: shl.b64
; CHECK: mov.b32
; CHECK: sub.s32
; CHECK: shr.u64
; CHECK: or.b64
; CHECK: add.s32
; CHECK: shl.b64
; CHECK: setp.gt.s32
; CHECK: selp.b64
; CHECK: shl.b64
  %amt = load i128, ptr %amtptr
  %a = load i128, ptr %val
  %val0 = shl i128 %a, %amt
  store i128 %val0, ptr %val
  ret void
}

; CHECK: shift_parts_right_128
define void @shift_parts_right_128(ptr %val, ptr %amtptr) {
; CHECK: shr.u64
; CHECK: sub.s32
; CHECK: shl.b64
; CHECK: or.b64
; CHECK: add.s32
; CHECK: shr.s64
; CHECK: setp.gt.s32
; CHECK: selp.b64
; CHECK: shr.s64
  %amt = load i128, ptr %amtptr
  %a = load i128, ptr %val
  %val0 = ashr i128 %a, %amt
  store i128 %val0, ptr %val
  ret void
}
