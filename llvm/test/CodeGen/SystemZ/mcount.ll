; Test proper insertion of mcount instrumentation
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -o - | FileCheck %s
;
; CHECK: # %bb.0:
; CHECK-NEXT: stg %r14, 8(%r15)
; CHECK-NEXT: brasl %r14, mcount@PLT
; CHECK-NEXT: lg %r14, 8(%r15)
define dso_local signext i32 @fib(i32 noundef signext %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %cmp = icmp sle i32 %0, 1
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  %1 = load i32, ptr %n.addr, align 4
  %sub = sub nsw i32 %1, 1
  %call = call signext i32 @fib(i32 noundef signext %sub)
  %2 = load i32, ptr %n.addr, align 4
  %sub1 = sub nsw i32 %2, 2
  %call2 = call signext i32 @fib(i32 noundef signext %sub1)
  %add = add nsw i32 %call, %call2
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ 1, %cond.true ], [ %add, %cond.false ]
  ret i32 %cond
}

attributes #0 = { "instrument-function-entry-inlined"="mcount" }
