; RUN: opt -S -loop-unswitch -instcombine -indvars -enable-new-pm=0 < %s | FileCheck %s

; This used to crash in SCEVExpander when there were congruent phis with and
; undef incoming value from the loop header. The -loop-unswitch -instcombine is
; necessary to create just this pattern, which is essentially a nop and gets
; folded away aggressively if spelled out in IR directly.
; PR 20093

@c = external global i32**, align 8

define void @test1() {
entry:
  br i1 undef, label %for.end12, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  %0 = load i32**, i32*** @c, align 8
  %1 = load i32*, i32** %0, align 8
  %2 = load i32, i32* %1, align 4
  br label %for.body

for.body:                                         ; preds = %for.cond.backedge, %for.body9.us, %for.cond.preheader
  %3 = phi i32* [ %1, %for.cond.preheader ], [ %3, %for.cond.backedge ], [ %6, %for.body9.us ]
  %4 = phi i32 [ %2, %for.cond.preheader ], [ undef, %for.cond.backedge ], [ %7, %for.body9.us ]
  %i.024 = phi i32 [ 0, %for.cond.preheader ], [ %inc, %for.cond.backedge ], [ 0, %for.body9.us ]
  %tobool1 = icmp eq i32 %4, 0
  br i1 %tobool1, label %if.end, label %for.cond.backedge

if.end:                                           ; preds = %for.body
  %5 = load i32, i32* %3, align 4
  %tobool4 = icmp eq i32 %5, 0
  br i1 %tobool4, label %for.cond3, label %for.body9.preheader

for.body9.preheader:                              ; preds = %if.end
  %tobool8 = icmp eq i32 %i.024, 1
  br i1 %tobool8, label %for.body9.us, label %for.body9

for.body9.us:                                     ; preds = %for.body9.preheader
  %6 = load i32*, i32** undef, align 8
  %7 = load i32, i32* %6, align 4
  br label %for.body

for.cond3:                                        ; preds = %for.cond3, %if.end
  br label %for.cond3

for.body9:                                        ; preds = %for.body9, %for.body9.preheader
  br label %for.body9

for.cond.backedge:                                ; preds = %for.body
  %inc = add nsw i32 %i.024, 1
  br i1 false, label %for.body, label %for.end12

for.end12:                                        ; preds = %for.cond.backedge, %entry
  ret void

; CHECK-LABEL: @test1
; CHECK-NOT: phi
}
