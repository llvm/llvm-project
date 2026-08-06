; RUN: opt < %s -disable-output -passes='no-op-module,loop-rotate,simplifycfg,hotcoldsplit'

; Check that we don't crash on deletion from a stale assumption cache.
;
; The no-op module pass initialises the assumption cache, while the SSA updater
; in `loop-rotate` invalidates it with a call to `Use::set()`. After
; simplifying the CFG, we're left with a `hotcoldsplit` candidate, leading to a
; call to `removeAffectedValues()` on a stale cache.

define void @dont_crash(i1 %cond.1) {
entry:
  %load = load ptr, ptr null, align 8
  br i1 %cond.1, label %loop.cond, label %assume

loop.cond:
  %phi = phi ptr [ null, %loop.body ], [ %load, %entry ]
  %cond.2 = phi i1 [ false, %loop.body ], [ true, %entry ]
  br i1 %cond.2, label %loop.body, label %dead

loop.body:
  call void @llvm.assume(i1 true) [ "nonnull"(ptr %phi) ]

  ; Lengthen code sequence so that `hotcoldsplit` fires (causing the deletion).
  call void @noise()
  call void @noise()
  call void @noise()
  call void @noise()

  br label %loop.cond

assume:
  call void @llvm.assume(i1 true) [ "dereferenceable"(ptr %load, i64 8) ]
  ret void

dead:
  unreachable
}

declare void @llvm.assume(i1 noundef)
declare void @noise()
