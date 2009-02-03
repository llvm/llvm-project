; RUN: llvm-as < %s | llc

; PR1224

declare i32 @test()
define i32 @test2() {
        %A = invoke i32 @test() to label %invcont unwind label %blat
invcont:
        ret i32 %A
blat:
        ret i32 0
}
