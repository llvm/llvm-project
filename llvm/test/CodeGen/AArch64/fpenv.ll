; RUN: llc -mtriple=aarch64-none-linux-gnu %s -o - | FileCheck %s

define void @func_02(i32 %rm) {
  call void @llvm.set.rounding(i32 %rm)
  ret void
}

; CHECK-LABEL: func_02:
; CHECK:       sub   w9, w0, #1
; CHECK:       mrs   x8, FPCR
; CHECK:       and   w9, w9, #0x3
; CHECK:       and   x8, x8, #0xffffffffff3fffff
; CHECK:       lsl   w9, w9, #22
; CHECK:       orr   x8, x8, x9
; CHECK:       msr   FPCR, x8
; CHECK:       ret


define void @func_03() {
  call void @llvm.set.rounding(i32 0)
  ret void
}

; CHECK-LABEL: func_03
; CHECK:       mrs   x8, FPCR
; CHECK:       orr   x8, x8, #0xc00000
; CHECK:       msr   FPCR, x8
; CHECK:       ret


define void @func_04() {
  call void @llvm.set.rounding(i32 1)
  ret void
}

; CHECK-LABEL: func_04
; CHECK:       mrs   x8, FPCR
; CHECK:       and   x8, x8, #0xffffffffff3fffff
; CHECK:       msr   FPCR, x8
; CHECK:       ret


define void @func_05() {
  call void @llvm.set.rounding(i32 2)
  ret void
}


; CHECK-LABEL: func_05
; CHECK:       mrs   x8, FPCR
; CHECK:       and   x8, x8, #0xffffffffff3fffff
; CHECK:       orr   x8, x8, #0x400000
; CHECK:       msr   FPCR, x8
; CHECK:       ret


define void @func_06() {
  call void @llvm.set.rounding(i32 3)
  ret void
}

; CHECK-LABEL: func_06
; CHECK:       mrs   x8, FPCR
; CHECK:       and   x8, x8, #0xffffffffff3fffff
; CHECK:       orr   x8, x8, #0x800000
; CHECK:       msr   FPCR, x8
; CHECK:       ret


declare void @llvm.set.rounding(i32)
