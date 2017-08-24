; RUN: llc < %s -mtriple=aarch64-none-eabi -mattr=+fullfp16 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ILLEGAL
; RUN: llc < %s -mtriple=aarch64-none-eabi -mattr=+fullfp16,+zcz | FileCheck %s --check-prefix=CHECK-ZCZ
; RUN: llc < %s -mtriple=aarch64-none-eabi -mattr=-fullfp16 | FileCheck %s --check-prefix=CHECK-NOFP16 --check-prefix=CHECK-ILLEGAL

define half @Const0() {
entry:
  ret half 0xH0000
}
; CHECK-DAG-ILLEGAL-LABEL:  Const0:
; CHECK-DAG-ILLEGAL-NEXT:   fmov  h0, wzr
; CHECK-DAG-ILLEGAL-NEXT:   ret

; CHECK-ZCZ-LABEL:  Const0:
; CHECK-ZCZ:        movi  v0.2d, #0000000000000000
; CHECK-ZCZ-NEXT:   ret

define half @Const1() {
entry:
  ret half 0xH3C00
}
; CHECK-DAG-LABEL: Const1:
; CHECK-DAG-NEXT:   fmov h0, #1.00000000
; CHECK-DAG-NEXT:   ret

; CHECK-NOFP16:        .[[LBL1:LCPI1_[0-9]]]:
; CHECK-NOFP16-NEXT:   .hword  15360 // half 1
; CHECK-NOFP16-LABEL:  Const1:
; CHECK-NOFP16:        adrp x[[NUM:[0-9]+]], .[[LBL1]]
; CHECK-NOFP16-NEXT:   ldr h0, [x[[NUM]], :lo12:.[[LBL1]]]

define half @Const2() {
entry:
  ret half 0xH3000
}
; CHECK-DAG-LABEL: Const2:
; CHECK-DAG-NEXT:   fmov h0, #0.12500000
; CHECK-DAG-NEXT:   ret

; CHECK-NOFP16:        .[[LBL2:LCPI2_[0-9]]]:
; CHECK-NOFP16-NEXT:   .hword  12288 // half 0.125
; CHECK-NOFP16-LABEL:  Const2:
; CHECK-NOFP16:        adrp x[[NUM:[0-9]+]], .[[LBL2]]
; CHECK-NOFP16-NEXT:   ldr h0, [x[[NUM]], :lo12:.[[LBL2]]]

define half @Const3() {
entry:
  ret half 0xH4F80
}
; CHECK-DAG-LABEL: Const3:
; CHECK-DAG-NEXT:   fmov h0, #30.00000000
; CHECK-DAG-NEXT:   ret

; CHECK-NOFP16:        .[[LBL3:LCPI3_[0-9]]]:
; CHECK-NOFP16-NEXT:   .hword  20352 // half 30
; CHECK-NOFP16-LABEL:  Const3:
; CHECK-NOFP16:        adrp x[[NUM:[0-9]+]], .[[LBL3]]
; CHECK-NOFP16-NEXT:   ldr h0, [x[[NUM]], :lo12:.[[LBL3]]]


define half @Const4() {
entry:
  ret half 0xH4FC0
}
; CHECK-DAG-LABEL: Const4:
; CHECK-DAG-NEXT:  fmov h0, #31.00000000
; CHECK-DAG-NEXT:  ret

; CHECK-NOFP16:        .[[LBL4:LCPI4_[0-9]]]:
; CHECK-NOFP16-NEXT:   .hword  20416                    // half 31
; CHECK-NOFP16-LABEL:  Const4:
; CHECK-NOFP16:        adrp x[[NUM:[0-9]+]], .[[LBL4]]
; CHECK-NOFP16-NEXT:   ldr h0, [x[[NUM]], :lo12:.[[LBL4]]]

define half @Const5() {
entry:
  ret half 0xH2FF0
}
; CHECK-ILLEGAL:        .[[LBL5:LCPI5_[0-9]]]:
; CHECK-ILLEGAL-NEXT:   .hword  12272                   // half 0.12402
; CHECK-ILLEGAL-LABEL:  Const5:
; CHECK-ILLEGAL:        adrp x[[NUM:[0-9]+]], .[[LBL5]]
; CHECK-ILLEGAL-NEXT:   ldr h0, [x[[NUM]], :lo12:.[[LBL5]]]

define half @Const6() {
entry:
  ret half 0xH4FC1
}
; CHECK-ILLEGAL:        .[[LBL6:LCPI6_[0-9]]]:
; CHECK-ILLEGAL-NEXT:   .hword  20417                   // half 31.016
; CHECK-ILLEGAL-LABEL:  Const6:
; CHECK-ILLEGAL:        adrp x[[NUM:[0-9]+]], .[[LBL6]]
; CHECK-ILLEGAL-NEXT:   ldr h0, [x[[NUM]], :lo12:.[[LBL6]]]


define half @Const7() {
entry:
  ret half 0xH5000
}
; CHECK-ILLEGAL:        .[[LBL7:LCPI7_[0-9]]]:
; CHECK-ILLEGAL-NEXT:   .hword  20480                   // half 32
; CHECK-ILLEGAL-LABEL:  Const7:
; CHECK-ILLEGAL:        adrp x[[NUM:[0-9]+]], .[[LBL7]]
; CHECK-ILLEGAL-NEXT:   ldr h0, [x[[NUM]], :lo12:.[[LBL7]]]


