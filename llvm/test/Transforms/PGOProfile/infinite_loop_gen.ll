; RUN: opt < %s -pgo-instr-gen -S -o -   | FileCheck %s

define void @foo() {
entry:
  br label %while.body
; CHECK: llvm.instrprof.increment

  while.body:                                       ; preds = %entry, %while.body
; CHECK: llvm.instrprof.increment
    call void (...) @bar() #2
    br label %while.body
}

declare void @bar(...)


attributes #0 = { nounwind }

