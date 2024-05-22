;; When BTI is enabled, keep the range check for a jump table for hardening,
;; even with an unreachable default.
;;
;; We check with and without the branch-target-enforcement module attribute,
;; and in each case, try overriding it with the opposite function attribute.
;; Expect to see a range check whenever there is BTI, and not where there
;; isn't.

; RUN: sed s/SPACE/4/ %s | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=BTI-TBB
; RUN: sed s/SPACE/4/ %s | sed '/test_jumptable/s/{/#0 {/' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=NOBTI-TBB
; RUN: sed s/SPACE/4/ %s | sed '/^..for-non-bti-build-sed-will-delete-everything-after-this-line/q' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=NOBTI-TBB
; RUN: sed s/SPACE/4/ %s | sed '/test_jumptable/s/{/#1 {/' | sed '/^..for-non-bti-build-sed-will-delete-everything-after-this-line/q' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=BTI-TBB

; RUN: sed s/SPACE/400/ %s | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=BTI-TBH
; RUN: sed s/SPACE/400/ %s | sed '/test_jumptable/s/{/#0 {/' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=NOBTI-TBH
; RUN: sed s/SPACE/400/ %s | sed '/^..for-non-bti-build-sed-will-delete-everything-after-this-line/q' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=NOBTI-TBH
; RUN: sed s/SPACE/400/ %s | sed '/test_jumptable/s/{/#1 {/' | sed '/^..for-non-bti-build-sed-will-delete-everything-after-this-line/q' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=BTI-TBH

; RUN: sed s/SPACE/400000/ %s | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=BTI-MOV
; RUN: sed s/SPACE/400000/ %s | sed '/test_jumptable/s/{/#0 {/' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=NOBTI-MOV
; RUN: sed s/SPACE/400000/ %s | sed '/^..for-non-bti-build-sed-will-delete-everything-after-this-line/q' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=NOBTI-MOV
; RUN: sed s/SPACE/400000/ %s | sed '/test_jumptable/s/{/#1 {/' | sed '/^..for-non-bti-build-sed-will-delete-everything-after-this-line/q' | llc -mtriple=thumbv8.1m.main-linux-gnu -mattr=+pacbti -o - | FileCheck %s --check-prefix=BTI-MOV

declare i32 @llvm.arm.space(i32, i32)

attributes #0 = { "branch-target-enforcement"="false" }
attributes #1 = { "branch-target-enforcement"="true"  }

define ptr @test_jumptable(ptr %src, ptr %dst) {
entry:
  %sw = load i32, ptr %src, align 4
  %src.postinc = getelementptr inbounds i32, ptr %src, i32 1
  switch i32 %sw, label %default [
    i32 0, label %sw.0
    i32 1, label %sw.1
    i32 2, label %sw.2
    i32 3, label %sw.3
  ]

sw.0:
  %store.0 = call i32 @llvm.arm.space(i32 SPACE, i32 14142)
  store i32 %store.0, ptr %dst, align 4
  br label %exit

sw.1:
  %store.1 = call i32 @llvm.arm.space(i32 SPACE, i32 31415)
  %dst.1 = getelementptr inbounds i32, ptr %dst, i32 1
  store i32 %store.1, ptr %dst.1, align 4
  br label %exit

sw.2:
  %store.2 = call i32 @llvm.arm.space(i32 SPACE, i32 27182)
  %dst.2 = getelementptr inbounds i32, ptr %dst, i32 2
  store i32 %store.2, ptr %dst.2, align 4
  br label %exit

sw.3:
  %store.3 = call i32 @llvm.arm.space(i32 SPACE, i32 16180)
  %dst.3 = getelementptr inbounds i32, ptr %dst, i32 3
  store i32 %store.3, ptr %dst.3, align 4
  br label %exit

default:
  unreachable

exit:
  ret ptr %src.postinc
}

; NOBTI-TBB:      test_jumptable:
; NOBTI-TBB-NEXT:         .fnstart
; NOBTI-TBB-NEXT: @ %bb
; NOBTI-TBB-NEXT:         ldr     [[INDEX:r[0-9]+]], [r0], #4
; NOBTI-TBB-NEXT: .LCPI
; NOBTI-TBB-NEXT:         tbb     [pc, [[INDEX]]]

; BTI-TBB:        test_jumptable:
; BTI-TBB-NEXT:           .fnstart
; BTI-TBB-NEXT:   @ %bb
; BTI-TBB-NEXT:           bti
; BTI-TBB-NEXT:           ldr     [[INDEX:r[0-9]+]], [r0], #4
; BTI-TBB-NEXT:           cmp     [[INDEX]], #3
; BTI-TBB-NEXT:           bhi     .LBB
; BTI-TBB-NEXT:   @ %bb
; BTI-TBB-NEXT:   .LCPI
; BTI-TBB-NEXT:           tbb     [pc, [[INDEX]]]

; NOBTI-TBH:      test_jumptable:
; NOBTI-TBH-NEXT:         .fnstart
; NOBTI-TBH-NEXT: @ %bb
; NOBTI-TBH-NEXT:         ldr     [[INDEX:r[0-9]+]], [r0], #4
; NOBTI-TBH-NEXT: .LCPI
; NOBTI-TBH-NEXT:         tbh     [pc, [[INDEX]], lsl #1]

; BTI-TBH:        test_jumptable:
; BTI-TBH-NEXT:           .fnstart
; BTI-TBH-NEXT:   @ %bb
; BTI-TBH-NEXT:           bti
; BTI-TBH-NEXT:           ldr     [[INDEX:r[0-9]+]], [r0], #4
; BTI-TBH-NEXT:           cmp     [[INDEX]], #3
; BTI-TBH-NEXT:           bhi.w   .LBB
; BTI-TBH-NEXT:   @ %bb
; BTI-TBH-NEXT:   .LCPI
; BTI-TBH-NEXT:           tbh     [pc, [[INDEX]], lsl #1]

; NOBTI-MOV:      test_jumptable:
; NOBTI-MOV-NEXT:         .fnstart
; NOBTI-MOV-NEXT: @ %bb
; NOBTI-MOV-NEXT:         ldr     [[INDEX:r[0-9]+]], [r0], #4
; NOBTI-MOV-NEXT:         adr.w   [[ADDR:r[0-9]+]], .LJTI
; NOBTI-MOV-NEXT:         add.w   [[ADDR]], [[ADDR]], [[INDEX]], lsl #2
; NOBTI-MOV-NEXT:         mov     pc, [[ADDR]]

; BTI-MOV:        test_jumptable:
; BTI-MOV-NEXT:           .fnstart
; BTI-MOV-NEXT:   @ %bb
; BTI-MOV-NEXT:           bti
; BTI-MOV-NEXT:           ldr     [[INDEX:r[0-9]+]], [r0], #4
; BTI-MOV-NEXT:           cmp     [[INDEX]], #3
; BTI-MOV-NEXT:           bls     .LBB
; BTI-MOV-NEXT:           b.w     .LBB
; BTI-MOV-NEXT:   .LBB
; BTI-MOV-NEXT:           adr.w   [[ADDR:r[0-9]+]], .LJTI
; BTI-MOV-NEXT:           add.w   [[ADDR]], [[ADDR]], [[INDEX]], lsl #2
; BTI-MOV-NEXT:           mov     pc, [[ADDR]]

; for-non-bti-build-sed-will-delete-everything-after-this-line
!llvm.module.flags = !{!0}
!0 = !{i32 8, !"branch-target-enforcement", i32 1}
