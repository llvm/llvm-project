; RUN: opt < %s -passes=lcssa,loop-simplify,indvars -S | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; PR28272, PR28825
; When LoopSimplify separates nested loops, it might break LCSSA form: values
; from the original loop might be used in the outer loop. This test invokes
; loop-unroll, which calls loop-simplify before itself. If LCSSA is broken
; after loop-simplify, we crash on assertion.

; CHECK-LABEL: @foo
define void @foo() {
entry:
  br label %header

header:
  br label %loop1

loop1:
  br i1 true, label %loop1, label %bb43

bb43:
  %a = phi i32 [ undef, %loop1 ], [ 0, %bb45 ], [ %a, %bb54 ]
  %b = phi i32 [ 0, %loop1 ], [ 1, %bb54 ], [ %c, %bb45 ]
  br i1 true, label %bb114, label %header

bb114:
  %c = add i32 0, 1
  %d = add i32 0, 1
  br i1 true, label %bb45, label %bb54

bb45:
  %x = add i32 %d, 0
  br label %bb43

bb54:
  br label %bb43
}

; CHECK-LABEL: @foo2
define void @foo2() {
entry:
  br label %outer

outer.loopexit:
  br label %outer

outer:
  br label %loop1

loop1:
  br i1 true, label %loop1, label %loop2.preheader

loop2.preheader:
  %a.ph = phi i32 [ undef, %loop1 ]
  %b.ph = phi i32 [ 0, %loop1 ]
  br label %loop2

loop2:
  %a = phi i32 [ 0, %loop2.if.true ], [ %a, %loop2.if.false ], [ %a.ph, %loop2.preheader ], [0, %bb]
  %b = phi i32 [ 1, %loop2.if.false ], [ %c, %loop2.if.true ], [ %b.ph, %loop2.preheader ], [%c, %bb]
  br i1 true, label %loop2.if, label %outer.loopexit

loop2.if:
  %c = add i32 0, 1
  switch i32 undef, label %loop2.if.false [i32 0, label %loop2.if.true
                                       i32 1, label %bb]

loop2.if.true:
  br i1 undef, label %loop2, label %bb

loop2.if.false:
  br label %loop2

bb:
  br label %loop2
}

; When LoopSimplify separates nested loops, it might break LCSSA form: values
; from the original loop might be used in exit blocks of the outer loop.
; CHECK-LABEL: @foo3
define void @foo3() {
entry:
  br label %bb1

bb1:
  br i1 undef, label %bb2, label %bb1

bb2:
  %a = phi i32 [ undef, %bb1 ], [ %a, %bb3 ], [ undef, %bb5 ]
  br i1 undef, label %bb3, label %bb1

bb3:
  %b = load ptr, ptr undef
  br i1 undef, label %bb2, label %bb4

bb4:
  br i1 undef, label %bb5, label %bb6

bb5:
  br i1 undef, label %bb2, label %bb4

bb6:
  br i1 undef, label %bb_end, label %bb1

bb_end:
  %x = getelementptr i32, ptr %b
  br label %bb_end
}

; When LoopSimplify separates nested loops, it might break LCSSA form: values
; from the original loop might occur in a loop, which is now a sibling of the
; original loop (before separating it was a subloop of the original loop, and
; thus didn't require an lcssa phi nodes).
; CHECK-LABEL: @foo4
define void @foo4() {
bb1:
  br label %bb2

; CHECK: bb2.loopexit:
bb2.loopexit:                                     ; preds = %bb3
  %i.ph = phi i32 [ 0, %bb3 ]
  br label %bb2

; CHECK: bb2.outer:
; CHECK: bb2:
bb2:                                              ; preds = %bb2.loopexit, %bb2, %bb1
  %i = phi i32 [ 0, %bb1 ], [ %i, %bb2 ], [ %i.ph, %bb2.loopexit ]
  %x = load i32, ptr undef, align 8
  br i1 undef, label %bb2, label %bb3.preheader

; CHECK: bb3.preheader:
bb3.preheader:                                    ; preds = %bb2
; CHECK: %x.lcssa = phi i32 [ %x, %bb2 ]
  br label %bb3

bb3:                                              ; preds = %bb3.preheader, %bb3
  %y = add i32 2, %x
  br i1 true, label %bb2.loopexit, label %bb3
}
