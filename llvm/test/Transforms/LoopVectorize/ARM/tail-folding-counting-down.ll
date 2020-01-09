; RUN: opt < %s -loop-vectorize -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -prefer-predicate-over-epilog -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -disable-mve-tail-predication=false -S | FileCheck %s

; Check that when we can't predicate this loop that it is still vectorised (with
; an epilogue).
; TODO: the reason this can't be predicated is because a primary induction
; variable can't be found (not yet) for this counting down loop. But with that
; fixed, this should be able to be predicated.

; CHECK-LABEL: vector.body:

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv8.1m.main-arm-unknown-eabihf"

define dso_local void @foo(i8* noalias nocapture readonly %A, i8* noalias nocapture readonly %B, i8* noalias nocapture %C, i32 %N) #0 {
entry:
  %cmp6 = icmp eq i32 %N, 0
  br i1 %cmp6, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %N.addr.010 = phi i32 [ %dec, %while.body ], [ %N, %while.body.preheader ]
  %C.addr.09 = phi i8* [ %incdec.ptr4, %while.body ], [ %C, %while.body.preheader ]
  %B.addr.08 = phi i8* [ %incdec.ptr1, %while.body ], [ %B, %while.body.preheader ]
  %A.addr.07 = phi i8* [ %incdec.ptr, %while.body ], [ %A, %while.body.preheader ]
  %incdec.ptr = getelementptr inbounds i8, i8* %A.addr.07, i32 1
  %0 = load i8, i8* %A.addr.07, align 1
  %incdec.ptr1 = getelementptr inbounds i8, i8* %B.addr.08, i32 1
  %1 = load i8, i8* %B.addr.08, align 1
  %add = add i8 %1, %0
  %incdec.ptr4 = getelementptr inbounds i8, i8* %C.addr.09, i32 1
  store i8 %add, i8* %C.addr.09, align 1
  %dec = add i32 %N.addr.010, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  ret void
}

attributes #0 = { nofree norecurse nounwind "target-features"="+armv8.1-m.main,+mve.fp" }
