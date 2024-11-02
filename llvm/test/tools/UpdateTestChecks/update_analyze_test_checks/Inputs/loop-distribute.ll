; RUN: opt -passes=loop-distribute -enable-loop-distribute \
; RUN:   -debug-only=loop-distribute -disable-output 2>&1 %s | FileCheck %s

define void @ldist(i1 %c, ptr %A, ptr %B, ptr %C) {
entry:
  br label %for.body

for.body:                                         ; preds = %if.end, %entry
  %iv = phi i16 [ 0, %entry ], [ %iv.next, %if.end ]
  %lv = load i16, ptr %A, align 1
  store i16 %lv, ptr %A, align 1
  br i1 %c, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %lv2 = load i16, ptr %A, align 1
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  %c.sink = phi ptr [ %B, %if.then ], [ %C, %for.body ]
  %lv3 = load i16, ptr %c.sink
  %iv.next = add nuw nsw i16 %iv, 1
  %tobool.not = icmp eq i16 %iv.next, 1000
  br i1 %tobool.not, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %if.end
  ret void
}
