; RUN: opt < %s -instcombine -S | grep "ret i32 1"

declare void @test2()

define i32 @test(i1 %cond, i32 *%P) {
  %A = alloca i32
  store i32 1, i32* %P
  store i32 1, i32* %A

  call void @test2() readonly

  %P2 = select i1 %cond, i32 *%P, i32* %A
  %V = load i32, i32* %P2
  ret i32 %V
}
