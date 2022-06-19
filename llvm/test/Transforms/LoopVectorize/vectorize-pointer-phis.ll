; RUN: opt < %s -passes=loop-vectorize -force-vector-width=2 -S | FileCheck %s

%s1 = type { [32000 x double], [32000 x double], [32000 x double] }

define i32 @load_with_pointer_phi_no_runtime_checks(%s1* %data) {
; CHECK-LABEL: @load_with_pointer_phi_no_runtime_checks
; CHECK-NOT: memcheck
; CHECK:     vector.body:
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp5 = icmp ult i64 %iv, 15999
  %arrayidx = getelementptr inbounds %s1, %s1 * %data, i64 0, i32 0, i64 %iv
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %loop.header
  %gep.1 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 1, i64 %iv
  br label %loop.latch

if.else:                                          ; preds = %loop.header
  %gep.2 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 2, i64 %iv
  br label %loop.latch

loop.latch:                                          ; preds = %if.else, %if.then
  %gep.2.sink = phi double* [ %gep.2, %if.else ], [ %gep.1, %if.then ]
  %v8 = load double, double* %gep.2.sink, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %arrayidx, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_no_runtime_checks(%s1* %data) {
; CHECK-LABEL: @store_with_pointer_phi_no_runtime_checks
; CHECK-NOT: memcheck
; CHECK:     vector.body
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp5 = icmp ult i64 %iv, 15999
  %arrayidx = getelementptr inbounds %s1, %s1 * %data, i64 0, i32 0, i64 %iv
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %loop.header
  %gep.1 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 1, i64 %iv
  br label %loop.latch

if.else:                                          ; preds = %loop.header
  %gep.2 = getelementptr inbounds %s1, %s1* %data, i64 0, i32 2, i64 %iv
  br label %loop.latch

loop.latch:                                          ; preds = %if.else, %if.then
  %gep.2.sink = phi double* [ %gep.2, %if.else ], [ %gep.1, %if.then ]
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %gep.2.sink, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_runtime_checks(double* %A, double* %B, double* %C) {
; CHECK-LABEL: @store_with_pointer_phi_runtime_checks
; CHECK:     memcheck
; CHECK:     vector.body
;
entry:
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp5 = icmp ult i64 %iv, 15999
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %loop.header
  %gep.1 = getelementptr inbounds double, double* %B, i64 %iv
  br label %loop.latch

if.else:                                          ; preds = %loop.header
  %gep.2 = getelementptr inbounds double, double* %C, i64 %iv
  br label %loop.latch

loop.latch:                                          ; preds = %if.else, %if.then
  %gep.2.sink = phi double* [ %gep.2, %if.else ], [ %gep.1, %if.then ]
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %gep.2.sink, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @load_with_pointer_phi_outside_loop(double* %A, double* %B, double* %C, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: @load_with_pointer_phi_outside_loop
; CHECK-NOT: vector.body
;
entry:
  br i1 %c.0, label %if.then, label %if.else

if.then:
  br label %loop.ph

if.else:
  %ptr.select = select i1 %c.1, double* %C, double* %B
  br label %loop.ph

loop.ph:
  %ptr = phi double* [ %A, %if.then ], [ %ptr.select, %if.else ]
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop.header ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %ptr, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %arrayidx, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}

define i32 @store_with_pointer_phi_outside_loop(double* %A, double* %B, double* %C, i1 %c.0, i1 %c.1) {
; CHECK-LABEL: @store_with_pointer_phi_outside_loop
; CHECK-NOT: vector.body
;
entry:
  br i1 %c.0, label %if.then, label %if.else

if.then:
  br label %loop.ph

if.else:
  %ptr.select = select i1 %c.1, double* %C, double* %B
  br label %loop.ph

loop.ph:
  %ptr = phi double* [ %A, %if.then ], [ %ptr.select, %if.else ]
  br label %loop.header

loop.header:                                        ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %loop.ph ], [ %iv.next, %loop.header ]
  %iv.next = add nuw nsw i64 %iv, 1
  %arrayidx = getelementptr inbounds double, double* %A, i64 %iv
  %v8 = load double, double* %arrayidx, align 8
  %mul16 = fmul double 3.0, %v8
  store double %mul16, double* %ptr, align 8
  %exitcond.not = icmp eq i64 %iv.next, 32000
  br i1 %exitcond.not, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i32 10
}
