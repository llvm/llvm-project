; RUN: opt -lint < %s

; lint shouldn't crash on any of the below functions

@g_1 = external global [3 x i32]
@g_2 = external global [2 x i32]

define void @test1() {
entry:
  tail call void @f1(i16 zext (i1 icmp eq (i32* getelementptr inbounds ([2 x i32], [2 x i32]* @g_2, i64 0, i64 0), i32* getelementptr inbounds ([3 x i32], [3 x i32]* @g_1, i64 0, i64 1)) to i16))
  ret void
}

declare void @f1(i16)

define void @test2() {
  tail call void inttoptr (i64 sext (i32 ptrtoint (void ()* @f2 to i32) to i64) to void ()*)()

  ret void
}

declare void @f2()

