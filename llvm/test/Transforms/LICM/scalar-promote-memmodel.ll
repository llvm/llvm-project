; RUN: opt < %s -basicaa -licm -S | FileCheck %s
; RUN: opt -aa-pipeline=type-based-aa,basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,loop(licm)' -S %s | FileCheck %s

; Make sure we don't hoist a conditionally-executed store out of the loop;
; it would violate the concurrency memory model

@g = common global i32 0, align 4

define void @bar(i32 %n, i32 %b) nounwind uwtable ssp {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc5, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tobool = icmp eq i32 %b, 0
  br i1 %tobool, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %tmp3 = load i32, i32* @g, align 4
  %inc = add nsw i32 %tmp3, 1
  store i32 %inc, i32* @g, align 4
  br label %for.inc

; CHECK: load i32, i32*
; CHECK-NEXT: add
; CHECK-NEXT: store i32

for.inc:                                          ; preds = %for.body, %if.then
  %inc5 = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
