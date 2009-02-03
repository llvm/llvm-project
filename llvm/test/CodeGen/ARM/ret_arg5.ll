; RUN: llvm-as < %s | llc -march=arm

define i32 @test(i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5) {
        ret i32 %a5
}
