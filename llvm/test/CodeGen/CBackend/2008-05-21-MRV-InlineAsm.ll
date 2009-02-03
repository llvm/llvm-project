; RUN: llvm-as < %s | llc -march=c

declare {i32, i32} @foo()

define i32 @test() {
  %A = call {i32, i32} @foo()
  %B = getresult {i32, i32} %A, 0
  %C = getresult {i32, i32} %A, 1
  %D = add i32 %B, %C
  ret i32 %D
}

define i32 @test2() {
  %A = call {i32, i32} asm sideeffect "...", "={cx},={di},~{dirflag},~{fpsr},~{flags},~{memory}"()
  %B = getresult {i32, i32} %A, 0
  %C = getresult {i32, i32} %A, 1
  %D = add i32 %B, %C
  ret i32 %D
}
