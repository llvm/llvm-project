; RUN: opt < %s -passes=lcssa -S -o - | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; PR28424
; Here LCSSA adds phi-nodes for %x into the loop exits. Then, SSAUpdater needs
; to insert phi-nodes to merge these values. That creates a new def, which in
; its turn needs another LCCSA phi-node, and this test ensures that we insert
; it.

; CHECK-LABEL: @foo1
define internal i32 @foo1(i1 %arg) {
entry:
  br label %header

header:
  %x = add i32 0, 1
  br i1 %arg, label %if, label %loopexit1

if:
  br i1 %arg, label %latch, label %loopexit2

latch:
  br i1 %arg, label %header, label %loopexit3

; CHECK: loopexit1:
; CHECK:   %x.lcssa = phi i32 [ %x, %header ]
loopexit1:
  br label %loop_with_insert_point

; CHECK: loopexit2:
; CHECK:   %x.lcssa1 = phi i32 [ %x, %if ]
loopexit2:
  br label %exit

; CHECK: loopexit3:
; CHECK:   %x.lcssa2 = phi i32 [ %x, %latch ]
loopexit3:
  br label %loop_with_insert_point

; CHECK: loop_with_insert_point:
; CHECK:   %x4 = phi i32 [ %x4, %loop_with_insert_point ], [ %x.lcssa2, %loopexit3 ], [ %x.lcssa, %loopexit1 ]
loop_with_insert_point:
  br i1 %arg, label %loop_with_insert_point, label %bb

; CHECK: bb:
; CHECK:   %x4.lcssa = phi i32 [ %x4, %loop_with_insert_point ]
bb:
  br label %exit

; CHECK: exit:
; CHECK:   %x3 = phi i32 [ %x4.lcssa, %bb ], [ %x.lcssa1, %loopexit2 ]
exit:
  ret i32 %x
}

; CHECK-LABEL: @foo2
define internal i32 @foo2(i1 %arg) {
entry:
  br label %header

header:
  %x = add i32 0, 1
  br i1 %arg, label %latch, label %loopexit1

latch:
  br i1 %arg, label %header, label %loopexit2

; CHECK: loopexit1:
; CHECK:   %x.lcssa = phi i32 [ %x, %header ]
loopexit1:
  br label %loop_with_insert_point

; CHECK: loopexit2:
; CHECK:   %x.lcssa1 = phi i32 [ %x, %latch ]
loopexit2:
  br label %loop_with_insert_point

; CHECK: loop_with_insert_point:
; CHECK:   %x2 = phi i32 [ %x2, %loop_with_insert_point ], [ %x.lcssa1, %loopexit2 ], [ %x.lcssa, %loopexit1 ]
loop_with_insert_point:
  br i1 %arg, label %loop_with_insert_point, label %exit

; CHECK: exit:
; CHECK:   %x2.lcssa = phi i32 [ %x2, %loop_with_insert_point ]
exit:
  ret i32 %x
}
