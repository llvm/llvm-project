; RUN: opt -passes=loop-deletion -verify-loop-info -verify-dom-info -disable-output < %s

; Regression test for llvm/llvm-project#244616.
;
; The loop latch %if.end6 ends in an indirectbr that targets the loop header
; %e.  %if.end6 has two successors and %e has two predecessors, so the
; backedge %if.end6 -> %e is a critical edge starting at an IndirectBrInst.
; SplitEdge() cannot split this edge, so LoopDeletion must leave the loop and
; LoopInfo intact. The verifier checks that the CFG and LoopInfo stay consistent.
;
; This is derived from the following non-UB C testcase:
;
;   int a = 0, b = 0;
;   void *c(void *p) {
;     static void *f[] = {&&g};
;
;     unsigned d = a;
;     while (1) {
;       if (__builtin_expect(0 >= d, 1))
;         break;
;       &&e;
;     e:
;       b++;
;       if (__builtin_expect(a, 0))
;         continue;
;       goto *p;
;     g:
;       d--;
;     }
;
;     return f[d];
;   }
;
; The IR is the module dumped immediately before LoopDeletionPass, with TBAA,
; branch-weight metadata, function attributes, and target/alignment decorations
; removed.  The control-flow shape is unchanged.

@a = global i32 0
@b = global i32 0

define ptr @c(ptr %p) {
entry:
  %0 = load i32, ptr @a
  %cmp15 = icmp eq i32 %0, 0
  br i1 %cmp15, label %while.end, label %e.preheader.lr.ph.lr.ph

e.preheader.lr.ph.lr.ph:
  %b.promoted = load i32, ptr @b
  br label %e.preheader.lr.ph

e.preheader.lr.ph:
  br label %e.preheader

while.body.loopexit:
  store i32 %1, ptr @b
  br i1 false, label %while.end.loopexit, label %e.preheader

e.preheader:
  %inc.lcssa1213 = phi i32 [ %b.promoted, %e.preheader.lr.ph ], [ %1, %while.body.loopexit ]
  %1 = add nsw i32 %inc.lcssa1213, 1
  br label %e

e:
  br i1 false, label %if.end6, label %while.body.loopexit

if.end6:
  indirectbr ptr %p, [label %g, label %e]

g:
  br i1 true, label %while.end.loopexit19, label %e.preheader.lr.ph

while.end.loopexit:
  br label %while.end

while.end.loopexit19:
  br label %while.end

while.end:
  ret ptr blockaddress(@c, %g)
}
