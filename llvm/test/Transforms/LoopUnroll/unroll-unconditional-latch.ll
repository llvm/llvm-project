; RUN: opt -loop-unroll -S %s -verify-loop-info -verify-dom-info -verify-loop-lcssa | FileCheck %s

%struct.spam = type { double, double, double, double, double, double, double }

define void @test2(i32* %arg)  {
; CHECK-LABEL: void @test2
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %for.header

; CHECK-LABEL: for.header:                                       ; preds = %entry
; CHECK-NEXT:    store i32 0, i32* %arg, align 4
; CHECK-NEXT:    br label %for.latch

; CHECK-LABEL: for.latch:                                        ; preds = %for.header
; CHECK-NEXT:    %ptr.1 = getelementptr inbounds i32, i32* %arg, i64 1
; CHECK-NEXT:    store i32 0, i32* %ptr.1, align 4
; CHECK-NEXT:    br label %for.latch.1

; CHECK-LABEL: if.end.loopexit:                                  ; preds = %for.latch.2
; CHECK-NEXT:    ret void

; CHECK-LABEL: for.latch.1:                                      ; preds = %for.latch
; CHECK-NEXT:    %ptr.2 = getelementptr inbounds i32, i32* %arg, i64 2
; CHECK-NEXT:    store i32 0, i32* %ptr.2, align 4
; CHECK-NEXT:    br label %for.latch.2

; CHECK-LABEL: for.latch.2:                                      ; preds = %for.latch.1
; CHECK-NEXT:    %ptr.3 = getelementptr inbounds i32, i32* %arg, i64 3
; CHECK-NEXT:    store i32 0, i32* %ptr.3, align 4
; CHECK-NEXT:    br i1 true, label %if.end.loopexit, label %for.latch.3

; CHECK-LABEL: for.latch.3:                                      ; preds = %for.latch.2
; CHECK-NEXT:    unreachable

entry:
  br label %for.header

for.header:                              ; preds = %for.latch, %entry
  %indvars.iv800 = phi i64 [ 0, %entry ], [ %indvars.iv.next801, %for.latch ]
  %ptr = getelementptr inbounds i32, i32* %arg, i64 %indvars.iv800
  store i32 0, i32* %ptr, align 4
  %indvars.iv.next801 = add nuw nsw i64 %indvars.iv800, 1
  %exitcond802 = icmp eq i64 %indvars.iv.next801, 4
  br i1 %exitcond802, label %if.end.loopexit, label %for.latch

for.latch: ; preds = %for.header
  br label %for.header

if.end.loopexit:                                  ; preds = %for.header
  ret void
}

define double @test_with_lcssa(double %arg1, double* %arg2) {
; CHECK-LABEL: define double @test_with_lcssa(
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %loop.header

; CHECK-LABEL: loop.header:                                      ; preds = %entry
; CHECK-NEXT:    %res = fsub double %arg1, 3.000000e+00
; CHECK-NEXT:    br label %loop.latch

; CHECK-LABEL: loop.latch:                                       ; preds = %loop.header
; CHECK-NEXT:    %ptr = getelementptr inbounds double, double* %arg2, i64 1
; CHECK-NEXT:    %lv = load double, double* %ptr, align 8
; CHECK-NEXT:    %res.1 = fsub double %lv, %res
; CHECK-NEXT:    br i1 true, label %loop.exit, label %loop.latch.1

; CHECK-LABEL: loop.exit:                                        ; preds = %loop.latch
; CHECK-NEXT:    %res.lcssa = phi double [ %res.1, %loop.latch ]
; CHECK-NEXT:    ret double %res.lcssa

; CHECK-LABEL: loop.latch.1:                                     ; preds = %loop.latch
; CHECK-NEXT:    %ptr.1 = getelementptr inbounds double, double* %arg2, i64 2
; CHECK-NEXT:    unreachable

entry:
  br label %loop.header

loop.header:                                            ; preds = %entry, %loop.latch
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %d1 = phi double [ %arg1, %entry ], [ %lv, %loop.latch ]
  %d2 = phi double [ 3.0, %entry ], [ %res, %loop.latch ]
  %res = fsub double %d1, %d2
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp eq i64 %iv.next, 2
  br i1 %cond, label %loop.exit, label %loop.latch

loop.latch:                                            ; preds = %bb366
  %ptr = getelementptr inbounds double, double* %arg2, i64 %iv.next
  %lv = load double, double* %ptr, align 8
  br label %loop.header

loop.exit:                                            ; preds = %bb366
  %res.lcssa = phi double [ %res, %loop.header ]
  ret double %res.lcssa
}

; We unroll the outer loop and need to preserve LI for the inner loop.
define void @test_with_nested_loop(i32* %arg)  {
; CHECK-LABEL: void @test_with_nested_loop
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %outer.header

; CHECK-DAG: outer.header:                                     ; preds = %entry
; CHECK-NEXT:    br label %inner.body.preheader

; CHECK-DAG: inner.body.preheader:                             ; preds = %outer.header
; CHECK-NEXT:    br label %inner.body

; CHECK-LABEL: inner.body:                                       ; preds = %inner.body.preheader, %inner.body
; CHECK-NEXT:    %j.iv = phi i64 [ %j.iv.next, %inner.body ], [ 0, %inner.body.preheader ]
; CHECK-NEXT:    %ptr = getelementptr inbounds i32, i32* %arg, i64 %j.iv
; CHECK-NEXT:    store i32 0, i32* %ptr, align 4
; CHECK-NEXT:    %j.iv.next = add nuw nsw i64 %j.iv, 1
; CHECK-NEXT:    %inner.cond = icmp eq i64 %j.iv.next, 40000
; CHECK-NEXT:    br i1 %inner.cond, label %outer.latch, label %inner.body

; CHECK-LABEL: outer.latch:                                      ; preds = %inner.body
; CHECK-NEXT:    br label %inner.body.preheader.1

; CHECK-LABEL: exit:                                             ; preds = %outer.latch.1
; CHECK-NEXT:    ret void

; CHECK-LABEL: inner.body.preheader.1:                           ; preds = %outer.latch
; CHECK-NEXT:    br label %inner.body.1

; CHECK-LABEL: inner.body.1:                                     ; preds = %inner.body.1, %inner.body.preheader.1
; CHECK-NEXT:    %j.iv.1 = phi i64 [ %j.iv.next.1, %inner.body.1 ], [ 0, %inner.body.preheader.1 ]
; CHECK-NEXT:    %idx.1 = add i64 1, %j.iv.1
; CHECK-NEXT:    %ptr.1 = getelementptr inbounds i32, i32* %arg, i64 %idx.1
; CHECK-NEXT:    store i32 0, i32* %ptr.1, align 4
; CHECK-NEXT:    %j.iv.next.1 = add nuw nsw i64 %j.iv.1, 1
; CHECK-NEXT:    %inner.cond.1 = icmp eq i64 %j.iv.next.1, 40000
; CHECK-NEXT:    br i1 %inner.cond.1, label %outer.latch.1, label %inner.body.1

; CHECK-LABEL: outer.latch.1:                                    ; preds = %inner.body.1
; CHECK-NEXT:    br i1 true, label %exit, label %inner.body.preheader.2

; CHECK-LABEL: inner.body.preheader.2:                           ; preds = %outer.latch.1
; CHECK-NEXT:    br label %inner.body.2

; CHECK-LABEL: inner.body.2:                                     ; preds = %inner.body.2, %inner.body.preheader.2
; CHECK-NEXT:    %j.iv.2 = phi i64 [ %j.iv.next.2, %inner.body.2 ], [ 0, %inner.body.preheader.2 ]
; CHECK-NEXT:    %idx.2 = add i64 2, %j.iv.2
; CHECK-NEXT:    %ptr.2 = getelementptr inbounds i32, i32* %arg, i64 %idx.2
; CHECK-NEXT:    store i32 0, i32* %ptr.2, align 4
; CHECK-NEXT:    %j.iv.next.2 = add nuw nsw i64 %j.iv.2, 1
; CHECK-NEXT:    %inner.cond.2 = icmp eq i64 %j.iv.next.2, 40000
; CHECK-NEXT:    br i1 %inner.cond.2, label %outer.latch.2, label %inner.body.2

; CHECK-LABEL: outer.latch.2:                                    ; preds = %inner.body.2
; CHECK-NEXT:    unreachable
;
entry:
  br label %outer.header

outer.header:                              ; preds = %outer.latch, %entry
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv, 2
  br i1 %outer.cond, label %exit, label %inner.body

inner.body:
  %j.iv = phi i64 [ 0, %outer.header ], [ %j.iv.next, %inner.body ]
  %idx = add i64 %outer.iv, %j.iv
  %ptr = getelementptr inbounds i32, i32* %arg, i64 %idx
  store i32 0, i32* %ptr, align 4
  %j.iv.next = add nuw nsw i64 %j.iv, 1
  %inner.cond = icmp eq i64 %j.iv.next, 40000
  br i1 %inner.cond, label %outer.latch, label %inner.body

outer.latch: ; preds = %inner.body
  br label %outer.header

exit:                                  ; preds = %outer.header
  ret void
}

; We unroll the inner loop and need to preserve LI for the outer loop.
define void @test_with_nested_loop_unroll_inner(i32* %arg)  {
; CHECK-LABEL: define void @test_with_nested_loop_unroll_inner(
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %outer.header

; CHECK-LABEL: outer.header:                                     ; preds = %inner.body, %entry
; CHECK-NEXT:   %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %inner.body ]
; CHECK-NEXT:   %outer.iv.next = add nuw nsw i64 %outer.iv, 1
; CHECK-NEXT:   %outer.cond = icmp eq i64 %outer.iv, 40000
; CHECK-NEXT:   br i1 %outer.cond, label %exit, label %inner.body.preheader

; CHECK-LABEL: inner.body.preheader:                             ; preds = %outer.header
; CHECK-NEXT:   br label %inner.body

; CHECK-LABEL: inner.body:                                       ; preds = %inner.body.preheader
; CHECK-NEXT:   %ptr = getelementptr inbounds i32, i32* %arg, i64 %outer.iv
; CHECK-NEXT:   store i32 0, i32* %ptr, align 4
; CHECK-NEXT:   %idx.1 = add i64 %outer.iv, 1
; CHECK-NEXT:   %ptr.1 = getelementptr inbounds i32, i32* %arg, i64 %idx.1
; CHECK-NEXT:   store i32 0, i32* %ptr.1, align 4
; CHECK-NEXT:   br label %outer.header

; CHECK-LABEL: exit:                                             ; preds = %outer.header
; CHECK-NEXT:  ret void
;
entry:
  br label %outer.header

outer.header:                              ; preds = %outer.latch, %entry
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv, 40000
  br i1 %outer.cond, label %exit, label %inner.body

inner.body:
  %j.iv = phi i64 [ 0, %outer.header ], [ %j.iv.next, %inner.body ]
  %idx = add i64 %outer.iv, %j.iv
  %ptr = getelementptr inbounds i32, i32* %arg, i64 %idx
  store i32 0, i32* %ptr, align 4
  %j.iv.next = add nuw nsw i64 %j.iv, 1
  %inner.cond = icmp eq i64 %j.iv.next, 2
  br i1 %inner.cond, label %outer.latch, label %inner.body

outer.latch: ; preds = %inner.body
  br label %outer.header

exit:                                  ; preds = %outer.header
  ret void
}



; Check that we do not crash for headers with non-branch instructions, e.g.
; switch. We do not unroll in those cases.
define void @test_switchinst_in_header() {
; CHECK-LABEL: define void @test_switchinst_in_header() {
; CHECK-LABEL: entry:
; CHECK-NEXT:    br label %while.header

; CHECK-LABEL: while.header:                                     ; preds = %while.latch, %entry
; CHECK-NEXT:    switch i32 undef, label %exit [
; CHECK-NEXT:      i32 11, label %while.body1
; CHECK-NEXT:      i32 5, label %while.body2
; CHECK-NEXT:    ]

; CHECK-LABEL: while.body1:                                      ; preds = %while.header
; CHECK-NEXT:    unreachable

; CHECK-LABEL: while.body2:                                      ; preds = %while.header
; CHECK-NEXT:    br label %while.latch

; CHECK-LABEL: while.latch:                                      ; preds = %while.body2
; CHECK-NEXT:    br label %while.header

; CHECK-LABEL: exit:                                             ; preds = %while.header
; CHECK-NEXT:    ret void
;
entry:
  br label %while.header

while.header:                               ; preds = %while.latch, %entry
  switch i32 undef, label %exit [
    i32 11, label %while.body1
    i32 5, label %while.body2
  ]

while.body1:                                ; preds = %while.header
  unreachable

while.body2:                                ; preds = %while.header
  br label %while.latch

while.latch:   								; preds = %while.body2
  br label %while.header

exit:                        			    ; preds = %while.header
  ret void
}
