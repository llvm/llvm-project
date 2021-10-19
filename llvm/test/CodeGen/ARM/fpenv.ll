; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+vfp2 %s -o - | FileCheck %s

define void @func_02(i32 %rm) {
  call void @llvm.set.rounding(i32 %rm)
  ret void
}

; CHECK-LABEL: func_02:
; CHECK:       vmrs  r1, fpscr
; CHECK:       sub   r0, r0, #1
; CHECK:       and   r0, r0, #3
; CHECK:       bic   r1, r1, #12582912
; CHECK:       orr   r0, r1, r0, lsl #22
; CHECK:       vmsr  fpscr, r0
; CHECK:       mov   pc, lr


define void @func_03() {
  call void @llvm.set.rounding(i32 0)
  ret void
}

; CHECK-LABEL: func_03
; CHECK:       vmrs  r0, fpscr
; CHECK:       orr   r0, r0, #12582912
; CHECK:       vmsr  fpscr, r0
; CHECK:       mov   pc, lr


define void @func_04() {
  call void @llvm.set.rounding(i32 1)
  ret void
}

; CHECK-LABEL: func_04
; CHECK:       vmrs    r0, fpscr
; CHECK:       bic     r0, r0, #12582912
; CHECK:       vmsr    fpscr, r0
; CHECK:       mov     pc, lr


define void @func_05() {
  call void @llvm.set.rounding(i32 2)
  ret void
}


; CHECK-LABEL: func_05
; CHECK:       vmrs    r0, fpscr
; CHECK:       bic     r0, r0, #12582912
; CHECK:       orr     r0, r0, #4194304
; CHECK:       vmsr    fpscr, r0
; CHECK:       mov     pc, lr


define void @func_06() {
  call void @llvm.set.rounding(i32 3)
  ret void
}

; CHECK-LABEL: func_06
; CHECK:       vmrs   r0, fpscr
; CHECK:       bic    r0, r0, #12582912
; CHECK:       orr    r0, r0, #8388608
; CHECK:       vmsr   fpscr, r0
; CHECK:       mov    pc, lr


declare void @llvm.set.rounding(i32)
