; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; Verify we do not crash for this test case.
;
; CHECK: polly.start
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @compressStream() #0 {
entry:
  br label %if.else

if.else:                                          ; preds = %entry
  br label %do.body.i

do.body.i:                                        ; preds = %for.cond.i.i.6, %for.cond.i.i.4, %do.body.i, %if.else
  %0 = phi i32 [ undef, %if.else ], [ 0, %for.cond.i.i.6 ], [ %div.i.i.2, %for.cond.i.i.4 ], [ %div.i.i.2, %do.body.i ]
  %add.i.i.2 = or i32 undef, undef
  %div.i.i.2 = udiv i32 %add.i.i.2, 10
  %1 = trunc i32 undef to i8
  %2 = icmp eq i8 %1, 0
  br i1 %2, label %for.cond.i.i.4, label %do.body.i

for.cond.i.i.4:                                   ; preds = %do.body.i
  br i1 undef, label %for.cond.i.i.6, label %do.body.i

for.cond.i.i.6:                                   ; preds = %for.cond.i.i.4
  br i1 undef, label %for.cond.i.i.7, label %do.body.i

for.cond.i.i.7:                                   ; preds = %for.cond.i.i.6
  unreachable
}
