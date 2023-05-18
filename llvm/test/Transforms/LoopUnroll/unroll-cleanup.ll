; PR23524
; The test is to check redundency produced by loop unroll pass
; should be cleaned up by later pass.
; RUN: opt < %s -O2 -S | FileCheck %s

; After loop unroll:
;       %dec18 = add nsw i32 %dec18.in, -1
;       ...
;       %dec18.1 = add nsw i32 %dec18, -1
; should be merged to:
;       %dec18.1 = add nsw i32 %dec18.in, -2
;
; CHECK-LABEL: @_Z3fn1v(
; CHECK: %dec18.1 = add nsw i32 %dec18.in, -2

; ModuleID = '<stdin>'
target triple = "x86_64-unknown-linux-gnu"

@b = global i32 0, align 4
@c = global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @_Z3fn1v() #0 {
entry:
  %tmp = load i32, ptr @b, align 4
  %tobool20 = icmp eq i32 %tmp, 0
  br i1 %tobool20, label %for.end6, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.cond1.for.cond.loopexit_crit_edge:            ; preds = %for.inc
  %add.ptr.lcssa = phi ptr [ %add.ptr, %for.inc ]
  %incdec.ptr.lcssa = phi ptr [ %incdec.ptr, %for.inc ]
  br label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.body, %for.cond1.for.cond.loopexit_crit_edge
  %r.1.lcssa = phi ptr [ %add.ptr.lcssa, %for.cond1.for.cond.loopexit_crit_edge ], [ %r.022, %for.body ]
  %a.1.lcssa = phi ptr [ %incdec.ptr.lcssa, %for.cond1.for.cond.loopexit_crit_edge ], [ %a.021, %for.body ]
  %tmp1 = load i32, ptr @b, align 4
  %tobool = icmp eq i32 %tmp1, 0
  br i1 %tobool, label %for.cond.for.end6_crit_edge, label %for.body

for.body:                                         ; preds = %for.cond.loopexit, %for.body.lr.ph
  %r.022 = phi ptr [ undef, %for.body.lr.ph ], [ %r.1.lcssa, %for.cond.loopexit ]
  %a.021 = phi ptr [ undef, %for.body.lr.ph ], [ %a.1.lcssa, %for.cond.loopexit ]
  %tmp2 = load i32, ptr @c, align 4
  %tobool215 = icmp eq i32 %tmp2, 0
  br i1 %tobool215, label %for.cond.loopexit, label %for.body3.lr.ph

for.body3.lr.ph:                                  ; preds = %for.body
  br label %for.body3

for.body3:                                        ; preds = %for.inc, %for.body3.lr.ph
  %dec18.in = phi i32 [ %tmp2, %for.body3.lr.ph ], [ %dec18, %for.inc ]
  %r.117 = phi ptr [ %r.022, %for.body3.lr.ph ], [ %add.ptr, %for.inc ]
  %a.116 = phi ptr [ %a.021, %for.body3.lr.ph ], [ %incdec.ptr, %for.inc ]
  %dec18 = add nsw i32 %dec18.in, -1
  %tmp3 = load i8, ptr %a.116, align 1
  %cmp = icmp eq i8 %tmp3, 0
  br i1 %cmp, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body3
  %arrayidx = getelementptr inbounds i16, ptr %r.117, i64 1
  store i16 0, ptr %arrayidx, align 2
  store i16 0, ptr %r.117, align 2
  %arrayidx5 = getelementptr inbounds i16, ptr %r.117, i64 2
  store i16 0, ptr %arrayidx5, align 2
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body3
  %incdec.ptr = getelementptr inbounds i8, ptr %a.116, i64 1
  %add.ptr = getelementptr inbounds i16, ptr %r.117, i64 3
  %tobool2 = icmp eq i32 %dec18, 0
  br i1 %tobool2, label %for.cond1.for.cond.loopexit_crit_edge, label %for.body3, !llvm.loop !0

for.cond.for.end6_crit_edge:                      ; preds = %for.cond.loopexit
  br label %for.end6

for.end6:                                         ; preds = %for.cond.for.end6_crit_edge, %entry
  ret void
}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.unroll.count", i32 2}
