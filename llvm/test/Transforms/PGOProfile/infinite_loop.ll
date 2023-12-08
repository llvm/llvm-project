; RUN: opt < %s -passes=instrprof -S -do-counter-promotion=1 | FileCheck %s
; CHECK: store

@__profn_foo = private constant [3 x i8] c"foo"

define void @foo() {
entry:
  br label %while.body

  while.body:                                       ; preds = %entry, %while.body
    call void @llvm.instrprof.increment(ptr @__profn_foo, i64 14813359968, i32 1, i32 0)
    call void (...) @bar() #2
    br label %while.body
}

declare void @bar(...)

declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #0

attributes #0 = { nounwind }

