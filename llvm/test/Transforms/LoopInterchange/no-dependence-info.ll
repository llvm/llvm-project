; RUN: opt %s -passes='loop-interchange' -pass-remarks=loop-interchange -disable-output 2>&1 | FileCheck --allow-empty %s

target triple = "aarch64-unknown-linux-gnu"

; CHECK-NOT: Computed dependence info, invoking the transform.

define dso_local void @_foo(ptr noundef %a, ptr noundef %neg, ptr noundef %pos) {
entry:
  %a.addr = alloca ptr, align 8
  %neg.addr = alloca ptr, align 8
  %pos.addr = alloca ptr, align 8
  %p = alloca i32, align 4
  %q = alloca i32, align 4
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %j = alloca i32, align 4
  store ptr %a, ptr %a.addr, align 8
  store ptr %neg, ptr %neg.addr, align 8
  store ptr %pos, ptr %pos.addr, align 8
  store i32 0, ptr %p, align 4
  store i32 0, ptr %q, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc16, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 32
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  store i32 2, ptr %cleanup.dest.slot, align 4
  br label %for.end18

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %1 = load i32, ptr %j, align 4
  %cmp2 = icmp slt i32 %1, 32
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.cond1
  store i32 5, ptr %cleanup.dest.slot, align 4
  br label %for.end

for.body4:                                        ; preds = %for.cond1
  %2 = load ptr, ptr %a.addr, align 8
  %3 = load i32, ptr %i, align 4
  %idxprom = sext i32 %3 to i64
  %arrayidx = getelementptr inbounds i32, ptr %2, i64 %idxprom
  %4 = load i32, ptr %arrayidx, align 4
  %cmp5 = icmp slt i32 %4, 0
  br i1 %cmp5, label %if.then, label %if.else

if.then:                                          ; preds = %for.body4
  %5 = load ptr, ptr %a.addr, align 8
  %6 = load i32, ptr %i, align 4
  %idxprom6 = sext i32 %6 to i64
  %arrayidx7 = getelementptr inbounds i32, ptr %5, i64 %idxprom6
  %7 = load i32, ptr %arrayidx7, align 4
  %8 = load ptr, ptr %neg.addr, align 8
  %9 = load i32, ptr %p, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, ptr %p, align 4
  %idxprom8 = sext i32 %9 to i64
  %arrayidx9 = getelementptr inbounds i32, ptr %8, i64 %idxprom8
  store i32 %7, ptr %arrayidx9, align 4
  br label %if.end

if.else:                                          ; preds = %for.body4
  %10 = load ptr, ptr %a.addr, align 8
  %11 = load i32, ptr %i, align 4
  %idxprom10 = sext i32 %11 to i64
  %arrayidx11 = getelementptr inbounds i32, ptr %10, i64 %idxprom10
  %12 = load i32, ptr %arrayidx11, align 4
  %13 = load ptr, ptr %pos.addr, align 8
  %14 = load i32, ptr %q, align 4
  %inc12 = add nsw i32 %14, 1
  store i32 %inc12, ptr %q, align 4
  %idxprom13 = sext i32 %14 to i64
  %arrayidx14 = getelementptr inbounds i32, ptr %13, i64 %idxprom13
  store i32 %12, ptr %arrayidx14, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %15 = load i32, ptr %j, align 4
  %inc15 = add nsw i32 %15, 1
  store i32 %inc15, ptr %j, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond.cleanup3
  br label %for.inc16

for.inc16:                                        ; preds = %for.end
  %16 = load i32, ptr %i, align 4
  %inc17 = add nsw i32 %16, 1
  store i32 %inc17, ptr %i, align 4
  br label %for.cond

for.end18:                                        ; preds = %for.cond.cleanup
  ret void
}

