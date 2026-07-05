; RUN: opt < %s -passes=instcombine -S | grep "ret i32 1"

declare void @test2()

define i32 @test(i1 %cond, ptr %P) {
  %A = alloca i32
  store i32 1, ptr %P
  store i32 1, ptr %A

  call void @test2() readonly

  %P2 = select i1 %cond, ptr %P, ptr %A
  %V = load i32, ptr %P2
  ret i32 %V
}
