;; Test cloning a single path with -basic-block-sections.

declare void @effect(i32 zeroext)

;; Test a valid application of path cloning.
; RUN: echo 'v1' > %t
; RUN: echo 'f foo' >> %t
; RUN: echo 'p 0 3 5' >> %t
; RUN: echo 'c 0 3.1 5.1 1 2 3 4 5' >> %t
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t -stop-after=bb-path-cloning | FileCheck %s --check-prefix=MIR

define void @foo(i1 %a, i1 %b, i1 %c, i1 %d) {
b0:
  call void @effect(i32 0)
  br i1 %a, label %b1, label %b3

b1:                                           ; preds = %b0
  call void @effect(i32 1)
  br i1 %b, label %b2, label %b3

b2:                                             ; preds = %b1
  call void @effect(i32 2)
  br label %b3

b3:                                            ; preds = %b0, %b1, %b2
  call void @effect(i32 3)
  br i1 %c, label %b4, label %b5

b4:                                             ; preds = %b3
  call void @effect(i32 4)
  br i1 %d, label %b5, label %cold

b5:                                            ; preds = %b3, %b4
  call void @effect(i32 5)
  ret void
cold:
  call void @effect(i32 6)                     ; preds = %b4
  ret void
}

;; Check the cloned block ids in MIR.

; MIR: bb.7.b3 (bb_id 3 1):
; MIR: bb.8.b5 (bb_id 5 1):

;; Check the final layout and branches.

;; bb section:
; CHECK:        .section    .text.foo,"ax",@progbits
; CHECK:      foo:
; CHECK:      # %bb.0:        # %b0
; CHECK:        jne .LBB0_1
; CHECK-NEXT: # %bb.7:        # %b3
; CHECK:        jne .LBB0_4
; CHECK-NEXT: # %bb.8:        # %b5
; CHECK:        retq
; CHECK-NEXT: .LBB0_1:        # %b1
; CHECK:        je .LBB0_3
; CHECK-NEXT: # %bb.2:        # %b2
; CHECK:        callq effect@PLT
; CHECK-NEXT: .LBB0_3:        # %b3
; CHECK:        je .LBB0_5
; CHECK-NEXT: .LBB0_4:        # %b4
; CHECK:        je foo.cold
; CHECK-NEXT: .LBB0_5:        # %b5
; CHECK:        retq

;; split section
; CHECK:        .section    .text.split.foo,"ax",@progbits
; CHECK:      foo.cold:      # %cold
