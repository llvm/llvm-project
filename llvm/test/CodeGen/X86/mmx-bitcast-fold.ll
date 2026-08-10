; RUN: opt -mtriple=x86_64-- -passes=early-cse -earlycse-debug-hash < %s -S | FileCheck %s

; CHECK: @foo(<1 x i64> zeroinitializer)

define void @bar() {
entry:
  %0 = bitcast double 0.0 to <1 x i64>
  %1 = call <1 x i64> @foo(<1 x i64> %0)
  ret void
}

declare <1 x i64> @foo(<1 x i64>)
