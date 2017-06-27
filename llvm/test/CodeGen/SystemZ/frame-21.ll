; Test the allocation of emergency spill slots.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; For frames of size less than 4096 - 2*160, no emercengy spill slot
; is required.  Check the maximum such case.
define void @f1(i64 %x) {
; CHECK-LABEL: f1:
; CHECK: stg %r2, 160(%r15)
; CHECK: br %r14
  %y = alloca [471 x i64], align 8
  %ptr = getelementptr inbounds [471 x i64], [471 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

; If the frame size is at least 4096 - 2*160, we do need the emergency
; spill slots.  Check the minimum such case.
define void @f2(i64 %x) {
; CHECK-LABEL: f2:
; CHECK: stg %r2, 176(%r15)
; CHECK: br %r14
  %y = alloca [472 x i64], align 8
  %ptr = getelementptr inbounds [472 x i64], [472 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

; However, if there are incoming stack arguments, those also need to be
; in reach, so the maximum frame size without emergency spill slots is
; 4096 - 2*160 - <size of incoming stack arguments>.  Check the maximum
; case where we still need no emergency spill slots ...
define void @f3(i64 %x, i64 %r3, i64 %r4, i64 %r5, i64 %r6, i64 %stack) {
; CHECK-LABEL: f3:
; CHECK: stg %r2, 160(%r15)
; CHECK: br %r14
  %y = alloca [470 x i64], align 8
  %ptr = getelementptr inbounds [470 x i64], [470 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

; ... and the minimum case where we do.
define void @f4(i64 %x, i64 %r3, i64 %r4, i64 %r5, i64 %r6, i64 %stack) {
; CHECK-LABEL: f4:
; CHECK: stg %r2, 176(%r15)
; CHECK: br %r14
  %y = alloca [471 x i64], align 8
  %ptr = getelementptr inbounds [471 x i64], [471 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

; Try again for the case of two stack arguments.
; Check the maximum case where we still need no emergency spill slots ...
define void @f5(i64 %x, i64 %r3, i64 %r4, i64 %r5, i64 %r6, i64 %stack1, i64 %stack2) {
; CHECK-LABEL: f5:
; CHECK: stg %r2, 160(%r15)
; CHECK: br %r14
  %y = alloca [469 x i64], align 8
  %ptr = getelementptr inbounds [469 x i64], [469 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

; ... and the minimum case where we do.
define void @f6(i64 %x, i64 %r3, i64 %r4, i64 %r5, i64 %r6, i64 %stack1, i64 %stack2) {
; CHECK-LABEL: f6:
; CHECK: stg %r2, 176(%r15)
; CHECK: br %r14
  %y = alloca [470 x i64], align 8
  %ptr = getelementptr inbounds [470 x i64], [470 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

