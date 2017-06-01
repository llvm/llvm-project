; RUN: opt -S -globalopt < %s | FileCheck %s

; CHECK: @tmp = local_unnamed_addr global i32 42

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]
@tmp = global i32 0

define i32 @TheAnswerToLifeTheUniverseAndEverything() {
  ret i32 42
}

define void @_GLOBAL__I_a() {
enter:
  %tmp1 = call i32 @TheAnswerToLifeTheUniverseAndEverything()
  store i32 %tmp1, i32* @tmp
  %cmp = icmp eq i32 %tmp1, 42
  call void @llvm.assume(i1 %cmp)
  ret void
}

declare void @llvm.assume(i1)
