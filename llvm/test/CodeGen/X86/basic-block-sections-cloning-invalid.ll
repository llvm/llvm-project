;; Tests for invalid or (partially invalid) path clonings with -basic-block-sections.

declare void @effect(i32 zeroext)

;; Test failed application of path cloning.
; RUN: echo 'v1' > %t1
; RUN: echo 'f foo' >> %t1
; RUN: echo 'p 0 2 3' >> %t1
; RUN: echo 'c 0 2.1 3.1 1' >> %t1
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t1 2> %t1.err | FileCheck %s
; RUN: FileCheck %s --check-prefixes=WARN1 < %t1.err
;; Test that valid clonings are applied correctly, even if invalid clonings exist.
; RUN: echo 'v1' > %t2
; RUN: echo 'f foo' >> %t2
; RUN: echo 'p 0 2 3' >> %t2
; RUN: echo 'p 0 1 3' >> %t2
; RUN: echo 'c 0 1.1 3.2 2.1 3.1 1' >> %t2
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t2 2> %t2.err | FileCheck %s --check-prefixes=PATH
; RUN: FileCheck %s --check-prefixes=WARN1 < %t2.err
; RUN: echo 'v1' > %t3
; RUN: echo 'f foo' >> %t3
; RUN: echo 'p 0 100' >> %t3
; RUN: echo 'c 0 100.1 1' >> %t3
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t3 2> %t3.err | FileCheck %s
; RUN: FileCheck %s --check-prefixes=WARN2 < %t3.err

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

; CHECK:   .section    .text.foo,"ax",@progbits
; CHECK: foo:
; CHECK: # %bb.0:        # %b0

; CHECK:   je .LBB0_3
; PATH:  # %bb.7:      # %b1
; PATH:  # %bb.8:      # %b3
; PATH:    jne .LBB0_4
; CHECK: # %bb.1:      # %b1
; CHECK:   jne foo.cold

; CHECK: foo.cold:      # %b2

;; Check the warnings
; WARN1: warning: block #2 is not a successor of block #0 in function foo
; WARN2: warning: no block with id 100 in function foo

