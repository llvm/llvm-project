; RUN: llvm-as < %s | llc -march=arm

define i32 @test(i32 %a1, i32 %a2) {
        ret i32 %a2
}

