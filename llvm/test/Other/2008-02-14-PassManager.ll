; RUN:  llvm-as < %s |  opt -loop-unroll -loop-rotate -simplifycfg -disable-output
; PR 2028
define i32 @test1() {
       ret i32 0;
}
