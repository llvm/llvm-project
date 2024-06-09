; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; CHECK-NOT: polly.start
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @f(i32 %argc, ptr %A) #0 {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %entry
  br label %for.end

for.end:                                          ; preds = %for.body, %entry
  %i.2 = phi i32 [ 1, %entry ], [ 2, %for.body ]
  %cmp170 = icmp eq i32 %i.2, %argc
  br i1 %cmp170, label %if.then172, label %if.end174

if.then172:                                       ; preds = %for.end
  %0 = load i32, ptr %A
  tail call void @usage()
  br label %if.end174

if.end174:                                        ; preds = %if.then172, %for.end
  %idxprom175 = sext i32 %i.2 to i64
  ret void
}

; Function Attrs: nounwind uwtable
declare void @usage()
