; RUN: opt < %s -scev-aa -aa-eval -print-all-alias-modref-info \
; RUN:   2>&1 | FileCheck %s

; At the time of this writing, -basicaa misses the example of the form
; A[i+(j+1)] != A[i+j], which can arise from multi-dimensional array references,
; and the example of the form A[0] != A[i+1], where i+1 is known to be positive.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"

; p[i] and p[i+1] don't alias.

; CHECK: Function: loop: 3 pointers, 0 call sites
; CHECK: NoAlias: double* %pi, double* %pi.next

define void @loop(double* nocapture %p, i64 %n) nounwind {
entry:
  %j = icmp sgt i64 %n, 0
  br i1 %j, label %bb, label %return

bb:
  %i = phi i64 [ 0, %entry ], [ %i.next, %bb ]
  %pi = getelementptr double* %p, i64 %i
  %i.next = add i64 %i, 1
  %pi.next = getelementptr double* %p, i64 %i.next
  %x = load double* %pi
  %y = load double* %pi.next
  %z = fmul double %x, %y
  store double %z, double* %pi
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Slightly more involved: p[j][i], p[j][i+1], and p[j+1][i] don't alias.

; CHECK: Function: nestedloop: 4 pointers, 0 call sites
; CHECK: NoAlias: double* %pi.j, double* %pi.next.j
; CHECK: NoAlias: double* %pi.j, double* %pi.j.next
; CHECK: NoAlias: double* %pi.j.next, double* %pi.next.j

define void @nestedloop(double* nocapture %p, i64 %m) nounwind {
entry:
  %k = icmp sgt i64 %m, 0
  br i1 %k, label %guard, label %return

guard:
  %l = icmp sgt i64 91, 0
  br i1 %l, label %outer.loop, label %return

outer.loop:
  %j = phi i64 [ 0, %guard ], [ %j.next, %outer.latch ]
  br label %bb

bb:
  %i = phi i64 [ 0, %outer.loop ], [ %i.next, %bb ]
  %i.next = add i64 %i, 1

  %e = add i64 %i, %j
  %pi.j = getelementptr double* %p, i64 %e
  %f = add i64 %i.next, %j
  %pi.next.j = getelementptr double* %p, i64 %f
  %x = load double* %pi.j
  %y = load double* %pi.next.j
  %z = fmul double %x, %y
  store double %z, double* %pi.j

  %o = add i64 %j, 91
  %g = add i64 %i, %o
  %pi.j.next = getelementptr double* %p, i64 %g
  %a = load double* %pi.j.next
  %b = fmul double %x, %a
  store double %b, double* %pi.j.next

  %exitcond = icmp eq i64 %i.next, 91
  br i1 %exitcond, label %outer.latch, label %bb

outer.latch:
  %j.next = add i64 %j, 91
  %h = icmp eq i64 %j.next, %m
  br i1 %h, label %return, label %outer.loop

return:
  ret void
}

; Even more involved: same as nestedloop, but with a variable extent.
; When n is 1, p[j+1][i] does alias p[j][i+1], and there's no way to
; prove whether n will be greater than 1, so that relation will always
; by MayAlias. The loop is guarded by a n > 0 test though, so
; p[j+1][i] and p[j][i] can theoretically be determined to be NoAlias,
; however the analysis currently doesn't do that.
; TODO: Make the analysis smarter and turn that MayAlias into a NoAlias.

; CHECK: Function: nestedloop_more: 4 pointers, 0 call sites
; CHECK: NoAlias: double* %pi.j, double* %pi.next.j
; CHECK: MayAlias: double* %pi.j, double* %pi.j.next

define void @nestedloop_more(double* nocapture %p, i64 %n, i64 %m) nounwind {
entry:
  %k = icmp sgt i64 %m, 0
  br i1 %k, label %guard, label %return

guard:
  %l = icmp sgt i64 %n, 0
  br i1 %l, label %outer.loop, label %return

outer.loop:
  %j = phi i64 [ 0, %guard ], [ %j.next, %outer.latch ]
  br label %bb

bb:
  %i = phi i64 [ 0, %outer.loop ], [ %i.next, %bb ]
  %i.next = add i64 %i, 1

  %e = add i64 %i, %j
  %pi.j = getelementptr double* %p, i64 %e
  %f = add i64 %i.next, %j
  %pi.next.j = getelementptr double* %p, i64 %f
  %x = load double* %pi.j
  %y = load double* %pi.next.j
  %z = fmul double %x, %y
  store double %z, double* %pi.j

  %o = add i64 %j, %n
  %g = add i64 %i, %o
  %pi.j.next = getelementptr double* %p, i64 %g
  %a = load double* %pi.j.next
  %b = fmul double %x, %a
  store double %b, double* %pi.j.next

  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %outer.latch, label %bb

outer.latch:
  %j.next = add i64 %j, %n
  %h = icmp eq i64 %j.next, %m
  br i1 %h, label %return, label %outer.loop

return:
  ret void
}

; ScalarEvolution expands field offsets into constants, which allows it to
; do aggressive analysis. Contrast this with BasicAA, which works by
; recognizing GEP idioms.

%struct.A = type { %struct.B, i32, i32 }
%struct.B = type { double }

; CHECK: Function: foo: 7 pointers, 0 call sites
; CHECK: NoAlias: %struct.B* %B, i32* %Z
; CHECK: NoAlias: %struct.B* %B, %struct.B* %C
; CHECK: MustAlias: %struct.B* %C, i32* %Z
; CHECK: NoAlias: %struct.B* %B, i32* %X
; CHECK: MustAlias: i32* %X, i32* %Z
; CHECK: MustAlias: %struct.B* %C, i32* %Y
; CHECK: MustAlias: i32* %X, i32* %Y

define void @foo() {
entry:
  %A = alloca %struct.A
  %B = getelementptr %struct.A* %A, i32 0, i32 0
  %Q = bitcast %struct.B* %B to %struct.A*
  %Z = getelementptr %struct.A* %Q, i32 0, i32 1
  %C = getelementptr %struct.B* %B, i32 1
  %X = bitcast %struct.B* %C to i32*
  %Y = getelementptr %struct.A* %A, i32 0, i32 1
  ret void
}

; CHECK: Function: bar: 7 pointers, 0 call sites
; CHECK: NoAlias: %struct.B* %N, i32* %P
; CHECK: NoAlias: %struct.B* %N, %struct.B* %R
; CHECK: MustAlias: %struct.B* %R, i32* %P
; CHECK: NoAlias: %struct.B* %N, i32* %W
; CHECK: MustAlias: i32* %P, i32* %W
; CHECK: MustAlias: %struct.B* %R, i32* %V
; CHECK: MustAlias: i32* %V, i32* %W

define void @bar() {
  %M = alloca %struct.A
  %N = getelementptr %struct.A* %M, i32 0, i32 0
  %O = bitcast %struct.B* %N to %struct.A*
  %P = getelementptr %struct.A* %O, i32 0, i32 1
  %R = getelementptr %struct.B* %N, i32 1
  %W = bitcast %struct.B* %R to i32*
  %V = getelementptr %struct.A* %M, i32 0, i32 1
  ret void
}

; CHECK: Function: nonnegative: 2 pointers, 0 call sites
; CHECK: NoAlias:  i64* %arrayidx, i64* %p

define void @nonnegative(i64* %p) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ] ; <i64> [#uses=2]
  %inc = add nsw i64 %i, 1                         ; <i64> [#uses=2]
  %arrayidx = getelementptr inbounds i64* %p, i64 %inc
  store i64 0, i64* %arrayidx
  %tmp6 = load i64* %p                            ; <i64> [#uses=1]
  %cmp = icmp slt i64 %inc, %tmp6                 ; <i1> [#uses=1]
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK: 14 no alias responses
; CHECK: 26 may alias responses
; CHECK: 18 must alias responses
