; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; CHECK-LABEL: for.cond:
; CHECK:         %num.0 = phi i32 [ %add, %for.body15 ], [ 0, %for.cond.pre_entry_bb ]
; CHECK:         br i1 false, label %for.body15, label %for.end22

; CHECK-LABEL: polly.merge_new_and_old:
; CHECK:         %num.0.merge = phi i32 [ %num.0.final_reload, %polly.exiting ], [ %num.0, %for.end22 ]
; CHECK:         br label %for.end44

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @func(ptr %p) {
entry:
  %counters = alloca [64 x i32], align 16
  %lenCounters = alloca [17 x i32], align 16
  br label %for.cond

for.cond:                                       ; preds = %for.body15, %for.cond
  %num.0 = phi i32 [ 0, %entry ], [ %add, %for.body15 ]
  br i1 false, label %for.body15, label %for.end22

for.body15:                                       ; preds = %for.cond
  %0 = load i32, ptr %counters, align 4
  %add = add i32 %num.0, %0
  br label %for.cond

for.end22:                                        ; preds = %for.cond
  br label %for.end44

for.end44:                                        ; preds = %for.end22
  br i1 undef, label %if.then50, label %if.end67

if.then50:                                        ; preds = %for.end44
  br label %cleanup

if.end67:                                         ; preds = %for.end44
  br label %do.body

do.body:                                          ; preds = %cond.end109, %if.end67
  %e.0 = phi i32 [ 0, %if.end67 ], [ %inc128, %cond.end109 ]
  br label %cond.end109

cond.end109:                                      ; preds = %do.body
  %idxprom122 = zext i32 %e.0 to i64
  %arrayidx123 = getelementptr inbounds i32, ptr %p, i64 %idxprom122
  %inc128 = add i32 %e.0, 1
  %sub129 = sub i32 %num.0, %inc128
  %cmp130 = icmp ugt i32 %sub129, 1
  br i1 %cmp130, label %do.body, label %do.end

do.end:                                           ; preds = %cond.end109
  %1 = load i32, ptr %arrayidx123, align 4
  %arrayidx142 = getelementptr inbounds [17 x i32], ptr %lenCounters, i64 0, i64 1
  store i32 2, ptr %arrayidx142, align 4
  br label %for.cond201

for.cond201:                                      ; preds = %for.body204, %do.end
  br i1 undef, label %for.body204, label %for.end214

for.body204:                                      ; preds = %for.cond201
  br label %for.cond201

for.end214:                                       ; preds = %for.cond201
  br label %cleanup

cleanup:                                          ; preds = %for.end214, %if.then50
  ret void
}
