; RUN: opt -S -passes=globalopt < %s | FileCheck %s

; CHECK: @tmp = local_unnamed_addr global i32 42

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_a, ptr null }]
@tmp = global i32 0

define i32 @TheAnswerToLifeTheUniverseAndEverything() {
  ret i32 42
}

define void @_GLOBAL__I_a() {
enter:
  %tmp1 = call i32 @TheAnswerToLifeTheUniverseAndEverything()
  store i32 %tmp1, ptr @tmp
  %cmp = icmp eq i32 %tmp1, 42
  call void @llvm.assume(i1 %cmp)
  ret void
}

declare void @llvm.assume(i1)
