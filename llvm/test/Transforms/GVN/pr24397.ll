; RUN: opt -passes=gvn -disable-output < %s

target triple = "x86_64-unknown-linux-gnu"

define i64 @foo(ptr %arrayidx) {
entry:
  %p = load ptr, ptr %arrayidx, align 8
  %cmpnull = icmp eq ptr %p, null
  br label %BB2

entry2:                                           ; No predecessors!
  br label %BB2

BB2:                                              ; preds = %entry2, %entry
  %load = load i64, ptr %arrayidx, align 8
  ret i64 %load
}
