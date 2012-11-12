; RUN: llc < %s -march=cpp
declare void @foo(<4 x i32>)
define void @bar() {
  call void @foo(<4 x i32> <i32 0, i32 1, i32 2, i32 3>)
  ret void
}
