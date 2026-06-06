;; Test for cloning a path ending with indirect branch with -basic-block-sections.

declare void @effect(i32 zeroext)

;; Test valid application of cloning for a path with indirect branch.
; RUN: echo 'v1' > %t
; RUN: echo 'f bar' >> %t
; RUN: echo 'p 0 1' >> %t
; RUN: echo 'c 0 1.1 2 1' >> %t
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t | FileCheck %s

define void @bar(i1 %a, i1 %b) {
b0:
  call void @effect(i32 0)
  br i1 %a, label %b1, label %b2
b1:                                              ; preds = %b0
  call void @effect(i32 1)
  %0 = select i1 %b,                           ; <ptr> [#uses=1]
              ptr blockaddress(@bar, %b2),
              ptr blockaddress(@bar, %b3)
  indirectbr ptr %0, [label %b2, label %b3]
b2:                                              ; preds = %b0, %b1
  call void @effect(i32 2)
  ret void
b3:
  call void @effect(i32 3)                       ; preds = %b1
  ret void
}

; CHECK:        .section    .text.bar,"ax",@progbits
; CHECK:      bar:
; CHECK:      # %bb.0:        # %b0
; CHECK:        je .LBB0_2
; CHECK-NEXT: # %bb.4:        # %b1
; CHECK:        jmpq *%rax
; CHECK-NEXT: .Ltmp0:         # Block address taken
; CHECK-NEXT: .LBB0_2:        # %b2
; CHECK:        retq
; CHECK-NEXT: # %bb.1:        # %b1
; CHECK:        jmpq *%rax
; CHECK:        .section    .text.split.bar,"ax",@progbits
; CHECK:      bar.cold:       # %b3

