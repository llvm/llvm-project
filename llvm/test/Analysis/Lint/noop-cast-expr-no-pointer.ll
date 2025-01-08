; RUN: opt -passes=lint < %s

; lint shouldn't crash on any of the below functions

@g_1 = external global [3 x i32]
@g_2 = external global [2 x i32]

define void @test1() {
entry:
  tail call void @f1(i1 ptrtoint (ptr @g_2 to i1))
  ret void
}

declare void @f1(i1)

define void @test2() {
  tail call void inttoptr (i64 -1 to ptr)()

  ret void
}

declare void @f2()

