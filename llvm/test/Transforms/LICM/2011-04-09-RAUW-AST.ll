; RUN: opt < %s -passes='loop-mssa(loop-rotate,licm)' -S | FileCheck %s
; PR9604

@g_3 = global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00"

define i32 @main() nounwind {
entry:
  %tmp = load i32, ptr @g_3, align 4
  %tobool = icmp eq i32 %tmp, 0
  br i1 %tobool, label %for.cond, label %if.then

if.then:                                          ; preds = %entry
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %if.then, %entry
  %g.0 = phi ptr [ %g.0, %for.inc10 ], [ @g_3, %entry ], [ null, %if.then ]
  %x.0 = phi i32 [ %inc12, %for.inc10 ], [ 0, %entry ], [ 0, %if.then ]
  %cmp = icmp slt i32 %x.0, 5
  br i1 %cmp, label %for.cond4, label %for.end13

for.cond4:                                        ; preds = %for.body7, %for.cond
  %y.0 = phi i32 [ %inc, %for.body7 ], [ 0, %for.cond ]
  %cmp6 = icmp slt i32 %y.0, 5
  br i1 %cmp6, label %for.body7, label %for.inc10

; CHECK: for.body7:
; CHECK-NEXT: phi
; CHECK-NEXT: store i32 0
; CHECK-NEXT: store i32 1

for.body7:                                        ; preds = %for.cond4
  store i32 0, ptr @g_3, align 4
  store i32 1, ptr %g.0, align 4
  %inc = add nsw i32 %y.0, 1
  br label %for.cond4

for.inc10:                                        ; preds = %for.cond4
  %inc12 = add nsw i32 %x.0, 1
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  %tmp14 = load i32, ptr @g_3, align 4
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %tmp14) nounwind
  ret i32 0
}

declare i32 @printf(ptr nocapture, ...) nounwind

