; RUN: opt -gvn-hoist -S < %s | FileCheck %s
; REQUIRES: asserts
%struct.job_pool.6.7 = type { i32 }

; CHECK-LABEL: @f()
define dso_local void @f() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  br label %for.body

for.body:                                         ; preds = %for.cond
  br label %if.end

if.then:                                          ; No predecessors!
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br i1 false, label %for.body12.lr.ph, label %for.end

for.body12.lr.ph:                                 ; preds = %if.end
  br label %for.body12

for.body12:                                       ; preds = %if.end40, %for.body12.lr.ph
  br label %if.then23

if.then23:                                        ; preds = %for.body12
  br i1 undef, label %if.then24, label %if.else

if.then24:                                        ; preds = %if.then23
  %0 = load ptr, ptr undef, align 8
  br label %if.end40

if.else:                                          ; preds = %if.then23
  %1 = load ptr, ptr undef, align 8
  br label %if.end40

if.end40:                                         ; preds = %if.else, %if.then24
  br i1 false, label %for.body12, label %for.cond9.for.end_crit_edge

for.cond9.for.end_crit_edge:                      ; preds = %if.end40
  br label %for.end

for.end:                                          ; preds = %for.cond9.for.end_crit_edge, %if.end
  br i1 true, label %if.then45, label %for.cond

if.then45:                                        ; preds = %for.end
  ret void
}
