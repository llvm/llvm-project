;; Check if manually reserved registers are always excluded from being saved by
;; the function prolog/epilog, even for callee-saved ones, as per GCC behavior.
;; X19(BP, LLVM specific), X29(FP), X30(LP), X31(SP) are special so
;; they are not checked.

; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu | FileCheck %s

define preserve_mostcc void @t1() "target-features"="+reserve-x1" {
; CHECK-LABEL: t1:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w1, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x1},{x1}"(i64 256)
  ret void
}

define preserve_mostcc void @t2() "target-features"="+reserve-x2" {
; CHECK-LABEL: t2:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w2, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x2},{x2}"(i64 256)
  ret void
}

define preserve_mostcc void @t3() "target-features"="+reserve-x3" {
; CHECK-LABEL: t3:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w3, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x3},{x3}"(i64 256)
  ret void
}

define preserve_mostcc void @t4() "target-features"="+reserve-x4" {
; CHECK-LABEL: t4:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w4, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x4},{x4}"(i64 256)
  ret void
}

define preserve_mostcc void @t5() "target-features"="+reserve-x5" {
; CHECK-LABEL: t5:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w5, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x5},{x5}"(i64 256)
  ret void
}

define preserve_mostcc void @t6() "target-features"="+reserve-x6" {
; CHECK-LABEL: t6:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w6, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x6},{x6}"(i64 256)
  ret void
}

define preserve_mostcc void @t7() "target-features"="+reserve-x7" {
; CHECK-LABEL: t7:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w7, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x7},{x7}"(i64 256)
  ret void
}

define preserve_mostcc void @t8() "target-features"="+reserve-x8" {
; CHECK-LABEL: t8:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w8, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x8},{x8}"(i64 256)
  ret void
}

define preserve_mostcc void @t9() "target-features"="+reserve-x9" {
; CHECK-LABEL: t9:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w9, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x9},{x9}"(i64 256)
  ret void
}

define preserve_mostcc void @t10() "target-features"="+reserve-x10" {
; CHECK-LABEL: t10:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w10, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x10},{x10}"(i64 256)
  ret void
}

define preserve_mostcc void @t11() "target-features"="+reserve-x11" {
; CHECK-LABEL: t11:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w11, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x11},{x11}"(i64 256)
  ret void
}

define preserve_mostcc void @t12() "target-features"="+reserve-x12" {
; CHECK-LABEL: t12:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w12, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x12},{x12}"(i64 256)
  ret void
}

define preserve_mostcc void @t13() "target-features"="+reserve-x13" {
; CHECK-LABEL: t13:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w13, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x13},{x13}"(i64 256)
  ret void
}

define preserve_mostcc void @t14() "target-features"="+reserve-x14" {
; CHECK-LABEL: t14:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w14, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x14},{x14}"(i64 256)
  ret void
}

define preserve_mostcc void @t15() "target-features"="+reserve-x15" {
; CHECK-LABEL: t15:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w15, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x15},{x15}"(i64 256)
  ret void
}

define preserve_mostcc void @t16() "target-features"="+reserve-x16" {
; CHECK-LABEL: t16:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w16, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x16},{x16}"(i64 256)
  ret void
}

define preserve_mostcc void @t17() "target-features"="+reserve-x17" {
; CHECK-LABEL: t17:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w17, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x17},{x17}"(i64 256)
  ret void
}

define preserve_mostcc void @t18() "target-features"="+reserve-x18" {
; CHECK-LABEL: t18:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w18, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x18},{x18}"(i64 256)
  ret void
}

define preserve_mostcc void @t20() "target-features"="+reserve-x20" {
; CHECK-LABEL: t20:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w20, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x20},{x20}"(i64 256)
  ret void
}

define preserve_mostcc void @t21() "target-features"="+reserve-x21" {
; CHECK-LABEL: t21:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w21, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x21},{x21}"(i64 256)
  ret void
}

define preserve_mostcc void @t22() "target-features"="+reserve-x22" {
; CHECK-LABEL: t22:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w22, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x22},{x22}"(i64 256)
  ret void
}

define preserve_mostcc void @t23() "target-features"="+reserve-x23" {
; CHECK-LABEL: t23:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w23, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x23},{x23}"(i64 256)
  ret void
}

define preserve_mostcc void @t24() "target-features"="+reserve-x24" {
; CHECK-LABEL: t24:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w24, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x24},{x24}"(i64 256)
  ret void
}

define preserve_mostcc void @t25() "target-features"="+reserve-x25" {
; CHECK-LABEL: t25:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w25, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x25},{x25}"(i64 256)
  ret void
}

define preserve_mostcc void @t26() "target-features"="+reserve-x26" {
; CHECK-LABEL: t26:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w26, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x26},{x26}"(i64 256)
  ret void
}

define preserve_mostcc void @t27() "target-features"="+reserve-x27" {
; CHECK-LABEL: t27:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w27, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x27},{x27}"(i64 256)
  ret void
}

define preserve_mostcc void @t28() "target-features"="+reserve-x28" {
; CHECK-LABEL: t28:
; CHECK: // %bb.0:
; CHECK-NEXT:        mov     w28, #256
; CHECK-NEXT:        //APP
; CHECK-NEXT:        //NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={x28},{x28}"(i64 256)
  ret void
}
