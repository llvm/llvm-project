; RUN: llvm-as %s -o /dev/null -f

define void @test(i32 %X) {
        call void @test( i32 6 )
        ret void
}
