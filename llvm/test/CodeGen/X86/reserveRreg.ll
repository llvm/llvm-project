;; Check if manually reserved registers are always excluded from being saved by
;; the function prolog/epilog, even for callee-saved ones, as per GCC behavior.

; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

define preserve_mostcc void @t8() "target-features"="+reserve-r8" {
; CHECK-LABEL: t8:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl	$256, %r8d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r8},{r8}"(i64 256)
  ret void
}

define preserve_mostcc void @t9() "target-features"="+reserve-r9" {
; CHECK-LABEL: t9:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl	$256, %r9d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r9},{r9}"(i64 256)
  ret void
}

define preserve_mostcc void @t10() "target-features"="+reserve-r10" {
; CHECK-LABEL: t10:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl	$256, %r10d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r10},{r10}"(i64 256)
  ret void
}

define preserve_mostcc void @t11() "target-features"="+reserve-r11" {
; CHECK-LABEL: t11:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl	$256, %r11d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r11},{r11}"(i64 256)
  ret void
}

define preserve_mostcc void @t12() "target-features"="+reserve-r12" {
; CHECK-LABEL: t12:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl	$256, %r12d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r12},{r12}"(i64 256)
  ret void
}

define preserve_mostcc void @t13() "target-features"="+reserve-r13" {
; CHECK-LABEL: t13:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl	$256, %r13d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r13},{r13}"(i64 256)
  ret void
}

define preserve_mostcc void @t14() "target-features"="+reserve-r14" {
; CHECK-LABEL: t14:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl   $256, %r14d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r14},{r14}"(i64 256)
  ret void
}

define preserve_mostcc void @t15() "target-features"="+reserve-r15" {
; CHECK-LABEL: t15:
; CHECK: # %bb.0:
; CHECK-NEXT:        movl   $256, %r15d
; CHECK-NEXT:        #APP
; CHECK-NEXT:        #NO_APP
; CHECK-NEXT:        ret
  call i64 asm sideeffect "", "={r15},{r15}"(i64 256)
  ret void
}

