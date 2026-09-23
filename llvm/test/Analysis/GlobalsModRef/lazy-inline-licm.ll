; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='cgscc(inline),require<globals-aa>,function(loop-mssa(licm))' -S | FileCheck %s
;
; LICM must not hoist a load across a call whose callee gained an unknown
; declaration call through inlining. After bar() is inlined into foo(),
; foo() directly calls the unknown declaration @qux(), so the load in @main
; must stay in the loop.
;
; Conversely, @decl_readnone() has no memory effects, so the load in @main2
; must still be hoisted out of the loop (precision check).
;
; The nounwind attributes are load-bearing: may-throw calls block motion
; regardless of AA.

@g = internal global i32 0
@H = internal global i32 0

declare void @qux()
declare void @opaque(ptr)
declare void @decl_readnone() readnone nounwind

define void @g_escape() {
  call void @opaque(ptr @g)
  ret void
}

define internal void @bar() {
  call void @qux()
  ret void
}

define void @foo() noinline nounwind {
  call void @bar()
  ret void
}

define i32 @main(i1 %c) {
; CHECK-LABEL: define i32 @main(
; CHECK-LABEL: loopA:
; CHECK: load i32, ptr @g
; CHECK-NEXT: call void @foo()
entry:
  br label %loopA
loopA:
  %v = load i32, ptr @g
  call void @foo()
  br i1 %c, label %loopA, label %exitA
exitA:
  %p = phi i32 [ %v, %loopA ]
  ret i32 %p
}

define internal void @bar2() {
  call void @decl_readnone()
  ret void
}

define void @foo2() noinline nounwind {
  call void @bar2()
  ret void
}

define i32 @main2(i1 %c) {
; CHECK-LABEL: define i32 @main2(
; CHECK-LABEL: loopB:
; CHECK-NOT: load i32, ptr @H
; CHECK: br i1 %c, label %loopB, label %exitB
entry:
  br label %loopB
loopB:
  %v = load i32, ptr @H
  call void @foo2()
  br i1 %c, label %loopB, label %exitB
exitB:
  %p = phi i32 [ %v, %loopB ]
  ret i32 %p
}
