;; Tests for invalid path cloning with -basic-block-sections involving indirect branches.

declare void @effect(i32 zeroext)

;; Test failed application of path cloning for paths with indirect branches.
; RUN: echo 'v1' > %t1
; RUN: echo 'f bar' >> %t1
; RUN: echo 'p 0 1 2' >> %t1
; RUN: echo 'c 0 1.1 2.1 1' >> %t1
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t1 2> %t1.err | FileCheck %s
; RUN: FileCheck %s --check-prefix=WARN < %t1.err
; RUN: echo 'v1' > %t2
; RUN: echo 'f bar' >> %t2
; RUN: echo 'p 1 2' >> %t2
; RUN: echo 'c 0 1 2.1' >> %t2
; RUN: llc < %s -mtriple=x86_64-pc-linux -O0 -function-sections -basic-block-sections=%t2 2> %t2.err | FileCheck %s
; RUN: FileCheck %s --check-prefix=WARN < %t2.err


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

; CHECK:   .section    .text.bar,"ax",@progbits
; CHECK:   bar:
; CHECK: # %bb.0:        # %b0
; CHECK: # %bb.1:        # %b1
; CHECK:   .section    .text.split.bar,"ax",@progbits
; CHECK: bar.cold:       # %b2   

; WARN: warning: block #1 has indirect branch and appears as the non-tail block of a path in function bar
