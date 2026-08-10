; RUN: opt -passes=ctx-prof-flatten %s -S | FileCheck %s

declare void @bar()

define void @foo() {
  call void @llvm.instrprof.increment(ptr @foo, i64 123, i32 1, i32 0)
  call void @llvm.instrprof.callsite(ptr @foo, i64 123, i32 1, i32 0, ptr @bar)
  call void @bar()
  ret void
}

; CHECK-NOT: call void @llvm.instrprof