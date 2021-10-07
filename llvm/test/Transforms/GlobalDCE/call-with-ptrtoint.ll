; RUN: opt < %s -globaldce -S

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)

define internal void @foo() {
  call void @bar_with_fptr_argument(i64 ptrtoint (void ()* @baz to i64))
  ret void
}

define internal void @bar_with_fptr_argument(i64 %0) {
  ret void
}

define internal void @baz() {
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
