; RUN: llc -mtriple=thumbv8 %s -o -       | FileCheck %s --check-prefixes=CHECK,T2
; RUN: llc -mtriple=thumbv8m.main %s -o - | FileCheck %s --check-prefixes=CHECK,T2
; RUN: llc -mtriple=thumbv8m.base %s -o - | FileCheck %s --check-prefixes=CHECK,T1
; RUN: llc -mtriple=thumbv7em %s -o -     | FileCheck %s --check-prefixes=CHECK,T2
; RUN: llc -mtriple=thumbv6m %s -o -      | FileCheck %s --check-prefixes=V6M

; Armv6m targets don't have a sdiv instruction, so sdiv should not appear at
; all in the output:

; V6M: .file {{.*}}
; V6M-NOT:  sdiv
; V6M-NOT:  idiv

; Test sdiv i16
define dso_local signext i16 @f0(i16 signext %F) local_unnamed_addr #0 {
; CHECK-LABEL: f0
; CHECK:       movs    r1, #2
; CHECK-NEXT:  sdiv    r0, r0, r1
; CHECK-NEXT:  sxth    r0, r0
; CHECK-NEXT:  bx      lr

entry:
  %0 = sdiv i16 %F, 2
  ret i16 %0
}

; Same as above, but now with i32
define dso_local i32 @f1(i32 %F) local_unnamed_addr #0 {
; CHECK-LABEL: f1
; CHECK:       movs    r1, #4
; CHECK-NEXT:  sdiv    r0, r0, r1
; CHECK-NEXT:  bx      lr

entry:
  %div = sdiv i32 %F, 4
  ret i32 %div
}

; The immediate is not a power of 2, so we expect a sdiv.
define dso_local i32 @f2(i32 %F) local_unnamed_addr #0 {
; CHECK-LABEL: f2
; CHECK:       movs    r1, #5
; CHECK-NEXT:  sdiv    r0, r0, r1
; CHECK-NEXT:  bx      lr

entry:
  %div = sdiv i32 %F, 5
  ret i32 %div
}

; Try a larger power of 2 immediate: immediates larger than
; 128 don't give any code size savings.
define dso_local i32 @f3(i32 %F) local_unnamed_addr #0 {
; CHECK-LABEL:  f3
; CHECK-NOT:    sdiv
entry:
  %div = sdiv i32 %F, 256
  ret i32 %div
}

attributes #0 = { minsize norecurse nounwind optsize readnone }


; These functions don't have the minsize attribute set, so should not lower
; the sdiv to sdiv, but to the faster instruction sequence.

define dso_local signext i16 @f4(i16 signext %F) {
; T2-LABEL:  f4
; T2:        uxth    r1, r0
; T2-NEXT:   add.w   r0, r0, r1, lsr #15
; T2-NEXT:   sxth    r0, r0
; T2-NEXT:   asrs    r0, r0, #1
; T2-NEXT:   bx      lr

; T1-LABEL: f4
; T1: 	    uxth  r1, r0
; T1-NEXT: 	lsrs  r1, r1, #15
; T1-NEXT: 	adds  r0, r0, r1
; T1-NEXT: 	sxth  r0, r0
; T1-NEXT: 	asrs  r0, r0, #1
; T1-NEXT: 	bx	lr

entry:
  %0 = sdiv i16 %F, 2
  ret i16 %0
}

define dso_local i32 @f5(i32 %F) {
; T2-LABEL: f5
; T2:       asrs  r1, r0, #31
; T2-NEXT:  add.w   r0, r0, r1, lsr #30
; T2-NEXT:  asrs    r0, r0, #2
; T2-NEXT:  bx      lr

; T1-LABEL: f5
; T1: 	    asrs r1, r0, #31
; T1-NEXT:	lsrs  r1, r1, #30
; T1-NEXT:	adds  r0, r0, r1
; T1-NEXT:	asrs  r0, r0, #2
; T1-NEXT:	bx  lr

entry:
  %div = sdiv i32 %F, 4
  ret i32 %div
}
