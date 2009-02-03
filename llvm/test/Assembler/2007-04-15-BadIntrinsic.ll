; RUN: not llvm-as %s -o /dev/null -f |& grep {Call to invalid LLVM intrinsic}

declare i32 @llvm.foobar(i32 %foo)

define i32 @test() {
  %nada = call i32 @llvm.foobar(i32 0)
  ret i32 %nada
}

