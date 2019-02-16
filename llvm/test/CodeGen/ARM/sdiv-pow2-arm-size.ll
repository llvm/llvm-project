; RUN: llc -mtriple=armv7a -mattr=+hwdiv-arm %s -o - | FileCheck %s --check-prefixes=CHECK,DIV
; RUN: llc -mtriple=armv7a -mattr=-hwdiv-arm %s -o - | FileCheck %s --check-prefixes=CHECK,NODIV

; Check SREM
define dso_local i32 @test_rem(i32 %F) local_unnamed_addr #0 {
; CHECK-LABEL: test_rem 
; CHECK:       asr   r1, r0, #31
; CHECK-NEXT:  add   r1, r0, r1, lsr #30
; CHECK-NEXT:  bic   r1, r1, #3
; CHECK-NEXT:  sub   r0, r0, r1

entry:
  %div = srem i32 %F, 4
  ret i32 %div
}

; Try an i16 sdiv, with a small immediate.
define dso_local signext i16 @f0(i16 signext %F) local_unnamed_addr #0 {
; CHECK-LABEL: f0

; DIV:         mov     r1, #2
; DIV-NEXT:    sdiv    r0, r0, r1
; DIV-NEXT:    sxth    r0, r0
; DIV-NEXT:    bx      lr

; NODIV:       uxth r1, r0
; NODIV-NEXT:  add  r0, r0, r1, lsr #15
; NODIV-NEXT:  sxth r0, r0
; NODIV-NEXT:  asr  r0, r0, #1
; NODIV-NEXT:  bx   lr

entry:
  %0 = sdiv i16 %F, 2
  ret i16 %0
}

; Try an i32 sdiv, with a small immediate.
define dso_local i32 @f1(i32 %F) local_unnamed_addr #0 {
; CHECK-LABEL: f1

; DIV:       mov     r1, #4
; DIV-NEXT:  sdiv    r0, r0, r1
; DIV-NEXT:  bx      lr

; NODIV:       asr  r1, r0, #31
; NODIV-NEXT:  add  r0, r0, r1, lsr #30
; NODIV-NEXT:  asr  r0, r0, #2
; NODIV-NEXT:  bx	  lr

entry:
  %div = sdiv i32 %F, 4
  ret i32 %div
}

; Try a large power of 2 immediate, which should also be materialised with 1
; move immediate instruction.
define dso_local i32 @f2(i32 %F) local_unnamed_addr #0 {
; CHECK-LABEL:  f2
; DIV:        mov r1, #131072
; DIV-NEXT:   sdiv  r0, r0, r1
; DIV-NEXT:   bx  lr
entry:
  %div = sdiv i32 %F, 131072
  ret i32 %div
}

; MinSize not set, so should expand to the faster but longer sequence.
define dso_local i32 @f3(i32 %F) {
; CHECK-LABEL: f3
; CHECK:       asr r1, r0, #31
; CHECK-NEXT:  add r0, r0, r1, lsr #30
; CHECK-NEXT:  asr r0, r0, #2
; CHECK-NEXT:  bx  lr
entry:
  %div = sdiv i32 %F, 4
  ret i32 %div
}

attributes #0 = { minsize norecurse nounwind optsize readnone }
