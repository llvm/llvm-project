;; Tests for path cloning with -basic-block-sections.

declare void @effect(i32 zeroext)

;; Test valid application of path cloning.
; RUN: echo 'v1' > %t1
; RUN: echo 'f foo' >> %t1
; RUN: echo 'p 0 3 5' >> %t1
; RUN: echo 'c 0 3.1 5.1 1 2 3 4 5' >> %t1
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -enable-basic-block-path-cloning -basic-block-sections=%t1 | FileCheck %s --check-prefixes=PATH1
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -enable-basic-block-path-cloning -basic-block-sections=%t1 -stop-after=bb-path-cloning | FileCheck %s --check-prefix=PATH1MIR
; RUN: echo 'v1' > %t2
; RUN: echo 'f foo' >> %t2
; RUN: echo 'p 0 3 5' >> %t2
; RUN: echo 'p 1 3 4 5' >> %t2
; RUN: echo 'c 0 3.1 5.1' >> %t2
; RUN: echo 'c 1 3.2 4.1 5.2 2 3 4 5' >> %t2
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -enable-basic-block-path-cloning -basic-block-sections=%t2 | FileCheck %s --check-prefixes=PATH2
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -enable-basic-block-path-cloning -basic-block-sections=%t2 -stop-after=bb-path-cloning | FileCheck %s --check-prefix=PATH2MIR

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

;; Check cloned block ids in MIR.

; PATH1MIR:   bb.7.b3 (bb_id 3 1):
; PATH1MIR    bb.8.b5 (bb_id 5 1):

; PATH2MIR:   bb.7.b3 (bb_id 3 1):
; PATH2MIR:   bb.8.b5 (bb_id 5 1):
; PATH2MIR:   bb.9.b3 (bb_id 3 2):
; PATH2MIR:   bb.10.b4 (bb_id 4 1):
; PATH2MIR:   bb.11.b5 (bb_id 5 2):


;; Check the final layout and branches.

; CHECK:        .section    .text.foo,"ax",@progbits
; CHECK:      foo:
; CHECK:      # %bb.0:        # %b0

; PATH1:        jne .LBB0_1
; PATH1-NEXT: # %bb.7:        # %b3
; PATH1:        jne .LBB0_4
; PATH1-NEXT: # %bb.8:        # %b5
; PATH1:        retq
; PATH1-NEXT: .LBB0_1:        # %b1
; PATH1:        je .LBB0_3
; PATH1-NEXT: # %bb.2:        # %b2
; PATH1:        callq effect@PLT
; PATH1-NEXT: .LBB0_3:        # %b3
; PATH1:        je .LBB0_5
; PATH1-NEXT: .LBB0_4:        # %b4
; PATH1:        je foo.cold
; PATH1-NEXT: .LBB0_5:        # %b5
; PATH1:        retq

; PATH2:        jne foo.__part.1
; PATH2-NEXT: # %bb.7:        # %b3
; PATH2:        jne .LBB0_4
; PATH2-NEXT: # %bb.8:        # %b5
; PATH2:        retq
; PATH2:        .section    .text.foo,"ax",@progbits,unique,1
; PATH2-NEXT: foo.__part.1:   # %b1
; PATH2:        jne .LBB0_2
; PATH2-NEXT: # %bb.9:        # %b3
; PATH2:        je .LBB0_5
; PATH2-NEXT: # %bb.10:       # %b4
; PATH2:        je foo.cold
; PATH2-NEXT: # %bb.11:       # %b5
; PATH2:        retq
; PATH2-NEXT: .LBB0_2:        # %b2
; PATH2:        callq	effect@PLT
; PATH2-NEXT: # %bb.3:        # %b3
; PATH2:        je .LBB0_5
; PATH2-NEXT: .LBB0_4:        # %b4
; PATH2:        je foo.cold
; PATH2-NEXT: .LBB0_5:       # %b5
; PATH2:        retq
; CHECK:        .section    .text.split.foo,"ax",@progbits
; CHECK:      foo.cold:      # %cold

; PATH3-WARN: warning: block #2 is not a successor of block #0 in function foo
; PATH4-WARN: warning: no block with id 100 in function foo

