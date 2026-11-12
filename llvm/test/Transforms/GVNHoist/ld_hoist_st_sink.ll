; Tests to make sure that loads and stores in a diamond get merged
; Loads are hoisted into the header. Stores sunks into the footer.
; RUN: opt -passes=gvn-hoist -S < %s | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

%struct.node = type { i64, ptr, ptr, ptr, i64, ptr, i64, i64, i64 }
%struct.arc = type { i64, i64, i64 }

define i64 @foo(ptr nocapture readonly %r) nounwind {
entry:
  %node.0.in16 = getelementptr inbounds %struct.node, ptr %r, i64 0, i32 2
  %node.017 = load ptr, ptr %node.0.in16, align 8
  %tobool18 = icmp eq ptr %node.017, null
  br i1 %tobool18, label %while.end, label %while.body.preheader

; CHECK-LABEL: while.body.preheader
while.body.preheader:                             ; preds = %entry
; CHECK: load
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %if.end
  %node.020 = phi ptr [ %node.0, %if.end ], [ %node.017, %while.body.preheader ]
  %sum.019 = phi i64 [ %inc, %if.end ], [ 0, %while.body.preheader ]
  %orientation = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 4
  %0 = load i64, ptr %orientation, align 8
  %cmp = icmp eq i64 %0, 1
  br i1 %cmp, label %if.then, label %if.else
; CHECK: if.then
if.then:                                          ; preds = %while.body
  %a = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 5
; CHECK-NOT: load %struct.arc
  %1 = load ptr, ptr %a, align 8
; CHECK-NOT: load i64, ptr
  %2 = load i64, ptr %1, align 8
  %pred = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 1
; CHECK-NOT: load ptr, ptr
  %3 = load ptr, ptr %pred, align 8
  %p = getelementptr inbounds %struct.node, ptr %3, i64 0, i32 6
; CHECK-NOT: load i64, ptr
  %4 = load i64, ptr %p, align 8
  %add = add nsw i64 %4, %2
  %p1 = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 6
; FIXME: store i64
  store i64 %add, ptr %p1, align 8
  br label %if.end

; CHECK: if.else
if.else:                                          ; preds = %while.body
  %pred2 = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 1
; CHECK-NOT: load ptr, ptr
  %5 = load ptr, ptr %pred2, align 8
  %p3 = getelementptr inbounds %struct.node, ptr %5, i64 0, i32 6
; CHECK-NOT: load i64, ptr
  %6 = load i64, ptr %p3, align 8
  %a4 = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 5
; CHECK-NOT: load ptr, ptr
  %7 = load ptr, ptr %a4, align 8
; CHECK-NOT: load i64, ptr
  %8 = load i64, ptr %7, align 8
  %sub = sub nsw i64 %6, %8
  %p6 = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 6
; FIXME: store i64
  store i64 %sub, ptr %p6, align 8
  br label %if.end

; CHECK: if.end
if.end:                                           ; preds = %if.else, %if.then
; FIXME: store
  %inc = add nsw i64 %sum.019, 1
  %node.0.in = getelementptr inbounds %struct.node, ptr %node.020, i64 0, i32 2
  %node.0 = load ptr, ptr %node.0.in, align 8
  %tobool = icmp eq ptr %node.0, null
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %if.end
  %inc.lcssa = phi i64 [ %inc, %if.end ]
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %sum.0.lcssa = phi i64 [ 0, %entry ], [ %inc.lcssa, %while.end.loopexit ]
  ret i64 %sum.0.lcssa
}
