; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Check that the stores are not hoisted: it is invalid to hoist stores if they
; are not executed on all paths. In this testcase, there are paths in the loop
; that do not execute the stores.

; CHECK-LABEL: define i32 @main
; CHECK: store
; CHECK: store
; CHECK: store

@a = global i32 0, align 4

define i32 @main() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %0 = load i32, i32* @a, align 4
  %cmp = icmp slt i32 %0, 1
  br i1 %cmp, label %for.cond1, label %for.end7

for.cond1:                                        ; preds = %for.cond, %for.inc
  %1 = load i32, i32* @a, align 4
  %cmp2 = icmp slt i32 %1, 1
  br i1 %cmp2, label %for.body3, label %for.inc5

for.body3:                                        ; preds = %for.cond1
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body3
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* @a, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3, %if.then
  %2 = load i32, i32* @a, align 4
  %inc4 = add nsw i32 %2, 1
  store i32 %inc4, i32* @a, align 4
  br label %for.cond1

for.inc5:                                         ; preds = %for.cond1
  %inc6 = add nsw i32 %1, 1
  store i32 %inc6, i32* @a, align 4
  br label %for.cond

for.end7:                                         ; preds = %for.cond
  ret i32 %0
}

