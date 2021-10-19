; RUN: opt -loop-unswitch -enable-new-pm=0 -verify-memoryssa -S - < %s | FileCheck %s

;CHECK-LABEL: @b
;CHECK: [[Loop1:for\.end.*]]:                              ; preds = %for.cond.us
;CHECK-NEXT:  %[[PhiVar1:pdt.*]] = phi i32 [ %pdt.0.us, %for.cond.us ]
;CHECK: [[Loop2:for\.end.*]]:                     ; preds = %for.cond.us1
;CHECK-NEXT:  %[[PhiVar2:pdt.*]] = phi i32 [ %pdt.0.us2, %for.cond.us1 ]
;CHECK: [[Loop3:for\.end.*]]:                        ; preds = %for.cond
;CHECK-NEXT:  %[[PhiVar3:pdt.*]] = phi i32 [ %pdt.0, %for.cond ]
;CHECK: [[Join1:for\.end.*]]:                                 ; preds = %[[Loop2]], %[[Loop3]]
;CHECK-NEXT:  %[[PhiRes1:pdt.*]] = phi i32 [ %[[PhiVar3]], %[[Loop3]] ], [ %[[PhiVar2]], %[[Loop2]] ]
;CHECK: for.end:                                          ; preds = %[[Loop1]], %[[Join1]]
;CHECK-NEXT:  %[[PhiRes2:pdt.*]] = phi i32 [ %[[PhiRes1]], %[[Join1]] ], [ %[[PhiVar1]], %[[Loop1]] ]
;CHECK-NEXT:  ret i32 %[[PhiRes2]]

; Function Attrs: nounwind uwtable
define i32 @b(i32 %x, i32 %y) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %pdt.0 = phi i32 [ 1, %entry ], [ %pdt.2, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tobool = icmp ne i32 %x, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %mul = mul nsw i32 %pdt.0, 2
  br label %if.end6

if.else:                                          ; preds = %for.body
  %tobool1 = icmp ne i32 %y, 0
  br i1 %tobool1, label %if.then2, label %if.else4

if.then2:                                         ; preds = %if.else
  %mul3 = mul nsw i32 %pdt.0, 3
  br label %if.end

if.else4:                                         ; preds = %if.else
  %mul5 = mul nsw i32 %pdt.0, 4
  br label %if.end

if.end:                                           ; preds = %if.else4, %if.then2
  %pdt.1 = phi i32 [ %mul3, %if.then2 ], [ %mul5, %if.else4 ]
  br label %if.end6

if.end6:                                          ; preds = %if.end, %if.then
  %pdt.2 = phi i32 [ %mul, %if.then ], [ %pdt.1, %if.end ]
  br label %for.inc

for.inc:                                          ; preds = %if.end6
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 %pdt.0
}

