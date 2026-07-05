; RUN: opt < %s -passes=loop-unroll,loop-rotate,simplifycfg -disable-output
; PR2028
define i32 @test1() {
       ret i32 0
}
