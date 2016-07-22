; RUN: opt < %s -lcssa -loop-unroll -S | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; PR28272
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
